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
from torch_sim.autobatching import calculate_memory_scalers, estimate_max_memory_scaler
from ._misc import ImplementationBase, DataSetIO
from .correlation import CorrelationCalculator

# ----------------------------------------------------------------------
#            Physical constants (SI) & Unit Conversions Value
# ----------------------------------------------------------------------
kB           = 1.380_649e-23                     # J·K⁻¹
ps_to_s      = 1e-12                             # Convertion value: picoseconds to seconds
eV_to_J      = 1.6021766208e-19                  # Convertion value: eV to Joules
Angs_to_m    = 1e-10                             # Convertion value: Angstrom to meter
amu_to_kg    = 1.660-27                          # Convertion value: amu to kilogram
k_conversion = eV_to_J / (Angs_to_m * ps_to_s)
# ----------------------------------------------------------------------

# --------------------------------------------------------------------
# Pending: (minor) Documentation and Better Comments + Logging info and variable name changes
# Homogenous Non-Equilibrium Molecular Dynamics (HNEMD) Method

# Citation: Zheyong Fan, Haikuan Dong, Ari Harju, and Tapio Ala-Nissila
# Homogeneous nonequilibrium molecular dynamics method for heat transport and spectral decomposition with many-body potentials
# Phys. Rev. B 99, 064308 (2019) DOI: 10.1103/PhysRevB.99.064308
# --------------------------------------------------------------------

import logging
logger = logging.getLogger(__name__)
class HNEMD(ImplementationBase, DataSetIO):
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
        super().__init__(method='HNEMD', logger=logger, system_state=system_state, temp_K=temp_K, timestep_ps=timestep_ps,
                         nsteps_total=nsteps_total, model=model, n_simulations=n_simulations,
                         model_memory_scaling=model_memory_scaling, perform_equilibration=perform_equilibration,
                         n_equilibration_steps=n_equilibration_steps, store_equilibration_data=store_equilibration_data,
                         equilibration_quantities_functions=equilibration_quantities_functions,
                         perform_geometry_optimization=perform_geometry_optimization, geometry_optimizer=geometry_optimizer,
                         optimizer_convergence_fn=optimizer_convergence_fn, data_folder_path_abs=data_folder_path_abs,
                         simulation_data_filename=simulation_data_filename,
                         equilibration_data_filename=equilibration_data_filename, display_pbar=display_pbar,
                         log_to_file=log_to_file)

    def run_simulation(
            self,
            Fe_vec: torch.Tensor,
            sample_interval: int,
            window_size: int,
            # batched_simulation: bool=False,
    ) -> None:
        self.Fe_vec = Fe_vec
        self.sample_interval = sample_interval
        self.window_size = window_size

        # Perform the geometry optimization:
        if self.perform_geometry_optimization:
            self.system_state = self._perform_geometry_optimization(self, logger)

        # Perform the equilibration steps
        if self.perform_equilibration:
            self.system_state = self._perform_equilibration(self, logger)

        self.corr_calc = [None] * self.n_simulations
        for i in range(self.n_simulations):
            corr_calc = CorrelationCalculator(
                window_size=window_size,
                properties={"hac": self._compute_heat_current},
                device=self.device,
                normalize=True,
            )
            self.corr_calc[i] = corr_calc

        self.corr_calc_count = 0
        # Setting the property recorder to only record crucial data during the simulation
        simulation_prop_to_record = {
            sample_interval: {
                "HAC": self._retrieve_hac,
            }
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
            autobatcher= self.binning_batcher, # True,
            pbar={'ascii': True, "desc": 'Simulation: ', 'colour': 'magenta'} if self.display_pbar else None,
        )
        logger.info("Simulation Completed.")

    def post_processing(self):
        factor = 1/(kB * self.temp_K)
        fe_mag = torch.norm(self.Fe_vec)
        fac = 1/(self.temp_K * fe_mag)
        timestep_s = self.simulation_timestep * ps_to_s
        for simulation_idx in range(len(self.simulation_filepath)):
            sim_filepath = self.simulation_filepath[simulation_idx]
            # Compute < J(t) >_ne (eqn 6)
            hac = self.load_property(sim_filepath, "HAC")
            J_ne = (factor * torch.cumulative_trapezoid(hac, dx=timestep_s*self.sample_interval, dim=0)) * self.Fe_vec

            # Compute k(t) (eqn 9)
            vol = torch.linalg.det(self.system_state.cell[simulation_idx])
            fac_scaled = fac/vol
            k = fac_scaled * torch.norm(J_ne, dim=-1) * k_conversion

            # Compute cumulative average k(t) (eqn 10)
            k_cumsum = torch.cumsum(k, dim=0)
            running_val = torch.arange(start=1, end=k.shape[0]+1, device=self.device)
            k_avg = k_cumsum/running_val

            # Pending logger output
            self.store_property(sim_filepath, 'k', k, self.nsteps_total)
            self.store_property(sim_filepath, 'k_avg', k_avg, self.nsteps_total)

        return None
    # ------------------------------------------------------------------------------------------------------------- #
    #                    Functions used to calculate simulation property during integration steps:
    # ------------------------------------------------------------------------------------------------------------- #
    def _retrieve_hac(self, state, *args):
        # Update the ith calculator based on the provided state
        calculator = self.corr_calc[self.corr_calc_count]
        calculator.update(state)

        # Retrieve the autocorrelation value if it exists or return dummy value
        if 'hac' in self.corr_calc[self.corr_calc_count].get_auto_correlations():
            # Get the latest value
            hac_value = self.corr_calc[self.corr_calc_count].get_auto_correlations()['hac'][-1, :]
        else:
            # Return dummy value
            hac_value = torch.zeros(size=(3,), device=self.device, dtype=self.dtype)

        # Update the counter
        if self.corr_calc_count+1 == self.n_simulations:
            self.corr_calc_count = 0
        else:
            self.corr_calc_count += 1
        return hac_value

    def _compute_heat_current(self, state):
        out = self.model(state)
        momenta = state.momenta           # shape: (n_atoms, 3)
        masses = state.masses             # shape: (n_atoms,)
        virials = out['virials']          # shape: (n_atoms, 3, 3)
        energy  = out['energies']

        # Calculate the Jq_k term
        momenta_norm = torch.norm(momenta, dim=1)
        momenta_squared = momenta_norm * momenta_norm
        denominator = 2 * torch.pow(masses,2)
        # Jq_k = sum(pi/mi * Ei)  where Ei = pi^2/2mi + U_i
        # t_1 = pi* |pi|^2/2m^2 |
        t1 = momenta * momenta_squared.unsqueeze(1) / denominator.unsqueeze(1)
        # t_2 = U_i * pi/mi
        t2 = energy.unsqueeze(1) * (momenta / masses.unsqueeze(-1))
        Jq_kinetic = torch.sum(t1+t2, dim=0)

        # Calculate the Jq_p term
        per_atom_heat_current_from_potential = torch.einsum('bmn,bn -> bm', virials, momenta/masses.unsqueeze(1)) # (n_atoms, 3)
        Jq_potential = torch.sum(per_atom_heat_current_from_potential, dim=0) # (3,)

        # Return the heat_current (MD units)
        return (Jq_kinetic + Jq_potential)
    # ------------------------------------ End of Simulation Prop Functions --------------------------------------- #

    @staticmethod
    def calculate_external_force(
            energies: torch.Tensor,
            momenta: torch.Tensor,
            masses: torch.Tensor,
            virials: torch.Tensor,
            Fe_vec: torch.Tensor
    )-> torch.Tensor:
        "F_ext or driving force for HNEMD"
        momenta_norm = torch.norm(momenta, dim=-1)
        momenta_squared = momenta_norm * momenta_norm
        term_1 = torch.einsum("b, n ->  bn", (0.5 * momenta_squared/masses + energies), Fe_vec)
        term_2 = torch.mean(term_1, dim=0)
        term_3 = torch.einsum("bmn,n->bm", virials, Fe_vec)
        term_4 = torch.mean(term_3, dim=0)
        return term_1 + term_3 - (term_2 + term_4)