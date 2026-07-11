import os
from os import PathLike
import sys
import warnings
from typing import Sequence, Optional, Tuple, Union, Mapping, Callable, Literal
import h5py
import numpy as np
import ase
import torch
import torch_sim as ts
from torch_sim.state import SimState
from torch_sim.integrators import MDState
from torch_sim.models.interface import ModelInterface
from torch_sim import fire_init, fire_step, gradient_descent_init, gradient_descent_step

from Feugmo_Group._misc import ImplementationBase, DataSetIO
from Feugmo_Group.correlation import CorrelationCalculator
# ----------------------------------------------------------------------
#            Physical constants (SI) & Unit Conversions Value
# ----------------------------------------------------------------------
kB          = 1.380_649e-23                     # J·K⁻¹
ps_to_s     = 1e-12                             # Convertion value: picoseconds to seconds
eV_to_J     = 1.6021766208e-19                  # Convertion value: eV to Joules
Angs_to_m   = 1e-10                             # Convertion value: Angstrom to meter
amu_to_kg   = 1.660-27                          # Convertion value: amu to kilogram
k_conversion = eV_to_J / (Angs_to_m * ps_to_s)  # Convertion value: k_md to k_si
n_conversion = eV_to_J * ps_to_s/(Angs_to_m**3) # Converstion value: n_md to n_si (n = Pa * s = J*s/m^3)
d_conversion = Angs_to_m**2 / ps_to_s           # Converstion value: d_md to d_si (d= m^2/s)
# ----------------------------------------------------------------------

# --------------------------------------------------------------------
# Pending: (minor) Documentation and Better Comments + Logging info and variable name changes

# Equilibrium Molecular Dynamics (EMD) Method based on Green-Kubo
# Citation:?
# --------------------------------------------------------------------

import logging
logger = logging.getLogger(__name__)

class EMD(ImplementationBase, DataSetIO):
    """

    """
    def __init__(
            self,
            # Simulation Parameters:
            system_state: SimState,
            temp_K: float,
            timestep_ps: float,
            nsteps_total: int,
            model: ModelInterface,
            n_simulations: int=5,
            model_memory_scaling: Literal["n_atoms", "n_atoms_x_density"]="n_atoms_x_density",
            # Equilibration Parameters:
            perform_equilibration: bool=True,
            n_equilibration_steps: int=5_000,
            store_equilibration_data: bool=True,
            equilibration_quantities_functions: Mapping[str, Callable]=None,
            # Geometry Optimization Parameters:
            perform_geometry_optimization: bool=False,
            geometry_optimizer: tuple[Callable, Callable] = (fire_init, fire_step),
            optimizer_convergence_fn: Callable=None,
            # Data and log files parameter:
            data_folder_path_abs: PathLike=None,
            simulation_data_filename: str='simulation_data',
            equilibration_data_filename: str='equilibrium_data',
            display_pbar: bool=False,
            log_to_file: bool=True,
    )-> None:
        super().__init__(method='EMD', logger=logger, system_state=system_state, temp_K=temp_K, timestep_ps=timestep_ps,
                         nsteps_total=nsteps_total, model=model, n_simulations=n_simulations,
                         model_memory_scaling=model_memory_scaling, perform_equilibration=perform_equilibration,
                         n_equilibration_steps=n_equilibration_steps, store_equilibration_data=store_equilibration_data,
                         equilibration_quantities_functions=equilibration_quantities_functions,
                         perform_geometry_optimization=perform_geometry_optimization, geometry_optimizer=geometry_optimizer,
                         optimizer_convergence_fn=optimizer_convergence_fn, data_folder_path_abs=data_folder_path_abs,
                         simulation_data_filename=simulation_data_filename,
                         equilibration_data_filename=equilibration_data_filename, display_pbar=display_pbar,
                         log_to_file=log_to_file)
        self.timestep_s = self.simulation_timestep * ps_to_s

    def run_simulation(
            self,
            # batched_simulation: bool=False,
            sample_interval: int,
            window_size: int,
            compute_thermal_conductivity: bool=False,
            compute_diffusivity: bool=False,
            compute_viscosity: bool=False, # Shear Viscosity
            compute_heat_using_energy_dependence: bool=None,
    ) -> None:
        self.sample_interval = sample_interval
        self.compute_thermal_conductivity = compute_thermal_conductivity
        self.compute_diffusivity = compute_diffusivity
        self.compute_viscosity = compute_viscosity
        self.window_size = window_size
        # Parameters used to calculate vacf using running average
        self._window_count = 0
        self._vacf_avg = torch.zeros(window_size, device=self.device)

        if self.nstep_total % sample_interval != 0:
            raise ValueError("Sample interval doesn't equally divide the total number of steps")
        if not (compute_viscosity or compute_diffusivity or compute_thermal_conductivity):
            raise ValueError('Specify at-least one transport coefficient for computation')
        if compute_heat_using_energy_dependence is not None and isinstance(compute_heat_using_energy_dependence, bool):
            self.compute_heat_using_energy_dependence = compute_heat_using_energy_dependence
        if self.nsteps_total % window_size != 0:
            raise ValueError("Window size doesn't equally divide the total number of steps")
        if compute_diffusivity and window_size * sample_interval > self.nsteps_total:
            raise ValueError("Window size * sample interval > nsteps_total. Diffusivity only stores the vacf "
                             "data when window is full i.e after every window_size*sample_interval steps")


        # Perform the geometry optimization:
        if self.perform_geometry_optimization:
            self.system_state = self._perform_geometry_optimization(self, logger)

        # Perform the equilibration steps
        if self.perform_equilibration:
            self.system_state = self._perform_equilibration(self, logger)

        # Implement the autocorrelation calc (WIP)
        props = (({} if not compute_thermal_conductivity else {'heat_current': self._compute_heat_current})
                 | ({} if not compute_viscosity else {'pressure': self._compute_pressure_tensor})
                 | ({} if not compute_diffusivity else {'velocities': lambda state: state.velocities}))
        self.corr_calc = [None] * self.n_simulations
        for i in range(self.n_simulations):
            corr_calc = CorrelationCalculator(
                window_size=window_size,
                properties=props,
                device=self.device,
                normalize=True,
            )
            self.corr_calc[i] = corr_calc

        self.corr_calc_count = 0

        # Setting the property recorder to only record crucial data during the simulation
        self.determine_updator_roles()
        simulation_prop_to_record = {
            sample_interval: (({'hac': self._retrieve_hac} if compute_thermal_conductivity else {})
                              | ({'pac': self._compute_viscosity} if compute_viscosity else {} )),
            (window_size*sample_interval): ({'vac': self._compute_diffusivity} if compute_diffusivity else {} ),
        }
        simulation_file_reporter =  ts.TrajectoryReporter(
            filenames=self.simulation_filepath,
            prop_calculators=simulation_prop_to_record,
        )

        self.system_state = ts.integrate(
            system = self.system_state,
            model  = self.model,
            integrator = ts.Integrator.nvt_nose_hoover,
            n_steps = self.nsteps_total,
            timestep = self.simulation_timestep,
            temperature = self.temp_K,
            trajectory_reporter = simulation_file_reporter,
            autobatcher=self.binning_batcher,
            pbar={'ascii': True, "desc": 'Simulation: ', 'colour': 'purple'} if self.display_pbar else None
        )
        logger.info("Simulation Completed.")

    def determine_updator_roles(self):
        things_to_compute = []
        map = {0: 'hac', 1: 'pac'}
        if self.compute_thermal_conductivity:
            things_to_compute.append(0)
        if self.compute_viscosity:
            things_to_compute.append(1)
        # if self.compute_diffusivity:
        #     things_to_compute.append(2)

        self.count_update_function = map[max(things_to_compute)]
        self.calc_update_function = map[min(things_to_compute)]

    def calc_updator(self, state):
        # Update the ith calculator based on the provided state
        self.corr_calc[self.corr_calc_count].update(state)

    def count_updator(self):
        # Update the counter
        if self.corr_calc_count + 1 == self.n_simulations:
            self.corr_calc_count = 0
        else:
            self.corr_calc_count += 1

    def post_processing(self):
        # Calculate the transport properties and change their units
        for i in range(self.n_simulations):
            vol = torch.linalg.det(self.system_state[i].cell)
            simulation_file = self.simulation_filepath[i]
            if self.compute_thermal_conductivity:
                # Shape: (nsteps_total / sample_interval + 1, 3) or (nsteps_total/sample_interval, 5)
                hac = self.load_property(simulation_file, 'hac')
                factor = 1 / (self.temp_K * self.temp_K * kB * vol)
                int_term = torch.cumulative_trapezoid(hac, dx=(self.timestep_s * self.sample_interval), dim=0)
                thermal_conductivity = factor * int_term * k_conversion
                self.store_property(simulation_file, 'thermal_conductivity', thermal_conductivity, self.nsteps_total)
                logger.info(f"Simulation {i} Results: K_μ: {torch.mean(thermal_conductivity).item():.4f} (W/mk), K_σ: {torch.std(thermal_conductivity).item():.4f} (W/mk)")
            if self.compute_viscosity:
                # Shape: (nsteps_total/sample_interval +1, 3)
                # n = V/Tk_B \int <Pab(x) Pab(0) > where Pab are off diagonal components of pressure tensor
                pac = self.load_property(self.simulation_file, 'pac')
                factor = vol / (kB * self.temp_K) # since we are using pressure tensor
                int_term = torch.cumulative_trapezoid(pac, dx=self.timestep_s*self.sample_interval, dim=0)
                viscosity = factor * int_term * n_conversion
                self.store_property(simulation_file, 'shear_viscosity', viscosity, self.nsteps_total)
                logger.info(f"Simulation {i} Results: n_μ: {torch.mean(viscosity).item():.4f} (Pa*s), n_σ: {torch.std(viscosity).item():.4f} (Pa*s)")
            if self.compute_diffusivity:
                # D = 1/3 \int <J(x) J(0) > where J is particle flux
                # unformatted vac dimensions: (nsteps_total/(window_size*sample_interval)+1, window_size) # +1 dummy value
                unformatted_vac = self.load_property(self.simulation_file, 'vac')
                number_entries = int(self.nsteps_total/(self.window_size*self.sample_interval)*self.window_size)
                vac = unformatted_vac[1:,:].reshape(number_entries) # dim (nsteps_total/sample_interval,)
                int_term = torch.cumulative_trapezoid(vac, dx=self.timestep_s*self.sample_interval, dim=0)
                diffusivity = 1/3 * int_term * d_conversion
                self.store_property(simulation_file, 'diffusivity', diffusivity, self.nsteps_total)
                logger.info(f"Simulation {i} Results: D_μ: {torch.mean(diffusivity).item():.4f} (m^2/s), D_σ: {torch.std(diffusivity).item():.4f} (m^2/s)")





    # ------------------------------------------------------------------------------------------------------------- #
    #                    Functions used to calculate simulation property during integration steps:
    # ------------------------------------------------------------------------------------------------------------- #

    def _compute_heat_current(self, state: MDState):
        mass = state.masses
        model_output = self.model(state)
        virials = model_output['virials']  # (Np, 3, 3)
        velocities = state.velocities

        if self.compute_heat_using_energy_dependence:
            potential_per_atom = model_output['energies']
            kinetic_energy = 0.5 * mass * torch.pow(torch.norm(velocities, dim=1), 2)
            energy = kinetic_energy + potential_per_atom  # (Np,)
            virials[:,0,0] += energy
            virials[:,1,1] += energy
            virials[:,2,2] += energy
            heat_flux = torch.einsum('nij,nj->ni', virials, velocities)
            heat_all = heat_flux.sum(dim=0)  # (3,)
        else:
            vx, vy, vz = velocities[:,0:1].squeeze(-1), velocities[:,1:2].squeeze(-1), velocities[:,2:].squeeze(-1)
            jx_in  = virials[:, 0, 0] * vx + virials[:, 0, 1] * vy                       # sigma_xx * vx + sigma_xy * vy
            jx_out = virials[:, 0, 2] * vz                                               # sigma_xz * vz
            jy_in  = virials[:, 1, 0] * vx + virials[:,1,1] * vy                         # sigma_yx * vx + sigma_yy * vy
            jy_out = virials[:, 1, 2] * vz                                               # sigma_yz * vz
            jz= virials[:, 2, 0] * vx + virials[:, 2, 1] * vy + virials[:, 2, 2] * vz    # sum(sigma_zi * vi) for i=(x,y,z)

            heat_all = torch.stack((jx_in.sum(), jx_out.sum(), jy_in.sum(), jy_out.sum(), jz.sum()))

        # (3,) or (5,) depending on the self.compute_heat_using_energy_dependence
        return heat_all * (eV_to_J/ps_to_s)

    def _retrieve_hac(self, state, *args):
        # Update the calculator (if req)
        if self.calc_update_function == 'hac':
            self.calc_updator(state)

        # Retrieve the latest/current HAC value
        if 'heat_current' in self.corr_calc[self.corr_calc_count].get_auto_correlations():
            hac = self.corr_calc[self.corr_calc_count].get_auto_correlations()['heat_current'][-1,:]
        else:
            # Return dummy value
            size = (3,) if not self.compute_heat_using_energy_dependence else (5,)
            hac = torch.zeros(size=size, device=self.device, dtype=self.dtype)

        # Update the counter (if req)
        if self.count_update_function == 'hac':
            self.count_updator()

        return hac

    def _compute_pressure_tensor(self, state: MDState):
        # Computing pressure tensor based on lammps documentation: https://docs.lammps.org/compute_pressure.html
        out = self.model(state)
        n_atoms = state.n_atoms
        vol = torch.linalg.det(state.cell) # I have no idea how to make this work for 2d
        pressure_tensor = (n_atoms*kB*self.temp_K)/vol + torch.sum(out['virials'], dim=0)
        # Off Diagonal triu components of pressure_tensor (3,3) -> (3,) "[Pxy, Pxz, Pyx]"
        return pressure_tensor[[0, 0, 1], [1, 2, 2]]

    def _retrieve_pac(self, state: MDState):
        # Update the calculator (if req)
        if self.calc_update_function == 'pac':
            self.calc_updator(state)

        # Retrieve the latest/current PAC value
        if 'pressure_tensor' in self.corr_calc[self.corr_calc_count].get_auto_correlations():
            pac = self.corr_calc[self.corr_calc_count].get_auto_correlations()['pressure_tensor'][-1,:]
        else:
            pac = torch.zeros(size=(3,), device=self.device, dtype=self.dtype)

        # Update the counter (if req)
        if self.count_update_function == 'pac':
            self.count_updator()
        return pac

    def _retrieve_vac(self, state):
        # Don't need to update the calculator

        if 'velocities' in self.corr_calc[self.corr_calc_count].get_auto_correlations():
            # Based on Torch_Sim's VelocityAutoCorrelation code
            vac = self.corr_calc[self.corr_calc_count].get_auto_correlations()['velocities']
            vacf = torch.mean(vac, dim=(1,2))
            self._window_count += 1
            factor = 1.0 / self._window_count
            self._vacf_avg += (vacf - self._vacf_avg) * factor

        # Update the counter (since it's working on a different freq)
        self.count_updator()
        # since vacf is torch_zeroes initially we can just send that dummy value
        return self._vacf_avg



    # ------------------------------------ End of Simulation Prop Functions --------------------------------------- #
