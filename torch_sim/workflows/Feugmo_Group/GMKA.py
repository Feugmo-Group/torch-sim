import os
from os import PathLike
import sys
from typing import Sequence, Optional, Tuple, Union, Mapping, Callable
import h5py
import numpy as np
import ase
import torch
from torch_sim.state import SimState
from torch_sim.integrators import MDState
from torch_sim.models.interface import ModelInterface
from torch_sim import fire_init, fire_step, gradient_descent_init, gradient_descent_step
import torch_sim as ts
from Feugmo_Group._misc import ImplementationBase, DataSetIO
from typing import Literal

# ----------------------------------------------------------------------
#            Physical constants (SI) & Unit Conversions Value
# ----------------------------------------------------------------------
kB          = 1.380_649e-23                     # J·K⁻¹
ps_to_s     = 1e-12                             # Convertion value: picoseconds to seconds
eV_to_J     = 1.6021766208e-19                  # Convertion value: eV to Joules
Angs_to_m   = 1e-10                             # Convertion value: Angstrom to meter
amu_to_kg   = 1.660-27                          # Convertion value: amu to kilogram
# ----------------------------------------------------------------------

# --------------------------------------------------------------------
# INCOMPLETE! (Doesn't work because we don't have eigenmodes and frequency calculation)
# Pending: (major, problematic) Figuring out a way to compute the dynamical matrix, probably have to create Hessian Matrix

# Green-Kubo Modal Analysis (GMKA) based on Wei Lv and Asegun Henry's Paper
# Citation:
# Wei Lv and Asegun Henry; Direct calculation of modal contributions to thermal conductivity via Green-Kubo modal analysis
# New J. Phys. 18, 013028 (2016); https://doi.org/10.1088/1367-2630/18/1/013028
# --------------------------------------------------------------------

import logging
from logging.handlers import RotatingFileHandler
logger = logging.getLogger(__name__)

class GMKA(ImplementationBase, DataSetIO):
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
        super().__init__(method='GKMA', logger=logger, system_state=system_state, temp_K=temp_K, timestep_ps=timestep_ps,
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
    ) -> None:
        self.sample_interval = sample_interval
        self.compute_thermal_conductivity = compute_thermal_conductivity

        if self.nstep_total % sample_interval != 0:
            raise Exception("Sample interval doesn't equally divide the total number of steps")
        if not (compute_viscosity or compute_diffusivity or compute_thermal_conductivity):
            raise ValueError('Specify at-least one transport coefficient for computation')
        if compute_heat_using_energy_dependence is not None and isinstance(compute_heat_using_energy_dependence, bool):
            self.compute_heat_using_energy_dependence = compute_heat_using_energy_dependence


        # Perform the geometry optimization:
        if self.perform_geometry_optimization:
            self.system_state = self._perform_geometry_optimization(self, logger)

        # Perform the equilibration steps
        if self.perform_equilibration:
            self.system_state = self._perform_equilibration(self, logger)

        # Implement the autocorrelation calc (WIP)
        self.corr_calc = [None] * self.n_simulations
        for i in range(self.n_simulations):
            corr_calc = ts.properties.correlations.CorrelationCalculator(
                window_size=window_size,
                properties={'heat_current': self._compute_heat_current}, # to be replaced with the actual function
                device=self.device,
                normalize=True,
            )
            self.corr_calc[i] = corr_calc

        self.corr_calc_count = 0

        # Setting the property recorder
        simulation_prop_to_record = {
            sample_interval: {'hac': self._retrieve_hac},
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

    def post_processing(self):
        # Calculate the transport properties and change their units
        for i in range(self.n_simulations):
            vol = torch.linalg.det(self.system_state[i].cell)
            simulation_file = self.simulation_filepath[i]
            if self.compute_thermal_conductivity:
                # Shape: (nsteps_total / sample_interval, 3) or (nsteps_total/sample_interval, 5)
                hac = self.load_property(simulation_file, 'hac')
                factor = 1 / (self.temp_K * self.temp_K * kB * vol)
                int_term = torch.cumulative_trapezoid(hac, dx=(self.timestep_s * self.sample_interval), dim=0)
                thermal_conductivity = factor * int_term * k_conversion
                self.store_property(simulation_file, 'thermal_conductivity', thermal_conductivity, self.nsteps_total)
                logger.info(f"Simulation {i} Results: K_μ: {torch.mean(rtc).item():.4f} (W/mk), K_σ: {torch.std(rtc).item():.4f} (W/mk)")


        # ------------------------------------------------------------------------------------------------------------- #
        #                    Functions used to calculate simulation property during integration steps:
        # ------------------------------------------------------------------------------------------------------------- #
        # Functions used to calculate the eigenmodes:
        # NOT THE CORRECT IMPLEMENTATION!
        # @staticmethod
        # def compute_dynamical_matrix_vectorized(velocities, masses, timestep, n_frames=None):
        #     """
        #     Compute dynamical matrix from MD trajectory using velocity autocorrelation.
        #     Fully vectorized—no Python loops.
        #
        #     Args:
        #         velocities: (T, N_atoms, 3) - velocity trajectory
        #         masses: (N_atoms,) - atomic masses
        #         timestep: MD timestep in fs
        #         n_frames: number of correlation frames (default: T//2)
        #
        #     Returns:
        #         D: (3*N_atoms, 3*N_atoms) - dynamical matrix
        #         frequencies: (3*N_atoms,) - eigenfrequencies in THz
        #     """
        #     T, N_atoms, _ = velocities.shape
        #     if n_frames is None:
        #         n_frames = T // 2
        #     device = velocities.device
        #
        #     # Flatten velocities: (T, 3*N_atoms)
        #     v_flat = velocities.reshape(T, -1)
        #
        #     # Compute velocity-velocity autocorrelation matrix
        #     # Using FFT-based correlation for speed
        #     # Pad to power of 2 for FFT efficiency
        #     T_padded = 2 ** torch.ceil(torch.log2(torch.tensor(2 * T))).int().item()
        #     v_padded = torch.zeros((T_padded, 3 * N_atoms), device=device, dtype=velocities.dtype)
        #     v_padded[:T] = v_flat
        #
        #     # FFT of velocities
        #     v_fft = torch.fft.fft(v_padded, dim=0)
        #
        #     # Power spectral density: |V(f)|^2
        #     psd = (v_fft * v_fft.conj()).real
        #
        #     # Inversing FFT to get autocorrelation
        #     autocorr_full = torch.fft.ifft(psd, dim=0).real
        #
        #     # Normalizing by number of time origins
        #     norm = torch.arange(T, 0, -1, device=device, dtype=velocities.dtype)
        #     autocorr = autocorr_full[:n_frames] / norm[:n_frames].unsqueeze(1)
        #
        #     # Compute dynamical matrix via Fourier transform of autocorrelation
        #     D = autocorr.sum(dim=0)
        #
        #     # Make symmetric
        #     D = (D + D.T) / 2
        #
        #     # Mass-weight: D_ij -> D_ij / (sqrt(m_i) * sqrt(m_j))
        #     mass_weights = torch.repeat_interleave(torch.sqrt(masses), 3)
        #     D = D / (mass_weights.unsqueeze(1) * mass_weights.unsqueeze(0))
        #
        #     # Eigendecompose
        #     eigenvals, eigenvecs = torch.linalg.eigh(D)
        #     frequencies = torch.sqrt(torch.clamp(eigenvals, min=1e-10))  # Frequencies (omega = sqrt(lambda))
        #     eigenvectors = eigenvecs.T  # Eigenvectors in mode-major order (M, 3*N_atoms)
        #
        #     return D, eigenvectors, frequencies


    def _retrieve_hac(self, state, *args):
        # Update the ith calculator based on the provided state
        self.corr_calc[self.corr_calc_count].update(state)

        # Retrieve the latest/current HAC value
        if 'heat_current' in self.corr_calc[self.corr_calc_count].correlations:
            hac = self.corr_calc[self.corr_calc_count].correlations['heat_current'][-1,:] # probably incorrect dimensions
        else:
            # Return dummy value
            # Unknown size
            hac = torch.zeros(size=size, device=self.device, dtype=self.dtype)

        # Update the counter
        if self.corr_calc_count + 1 == self.n_simulations:
            self.corr_calc_count = 0
        else:
            self.corr_calc_count += 1

        return hac

    # ------------------------------------ End of Simulation Prop Functions --------------------------------------- #



    # ------------------------------------------------------------------------------------------------------------- #
    #                    Functions used to calculate simulation property during integration steps:
    # ------------------------------------------------------------------------------------------------------------- #

    def _calculate_heat_per_atom(self, state: MDState, model: ModelInterface):
        # Code if we had the eigenvalues (@ operations will probably raise an error in actual sim)
        device = self.device
        virials = self.model(state)['virials']
        vx, vy, vz = state.velocities[:,0:1].detach(), state.velocities[:,1:2].detach(), state.velocities[:,2:].detach()
        sqrtmass = torch.sqrt(state.masses)
        rsqrtmass = 1/sqrtmass

        # mass-weighted velocities
        mvx = vx * sqrtmass
        mvy = vy * sqrtmass
        mvz = vz * sqrtmass

        # modal velocities
        eigx, eigy, eigz = None, None, None # to be replaced
        M = eigx.shape[0]
        xdotx = eigx @ mvx
        xdoty = eigy @ mvy
        xdotz = eigz @ mvz

        # sm construction (scale virial by rsqrtmass)
        sm = virials * rsqrtmass[:, None, None]
        smx = sm[:,:,0]
        smy = sm[:,:,1]
        smz = sm[:,:,2]

        # intermediate matrices
        jmx = eigx @ smx
        jmy = eigy @ smy
        jmz = eigz @ smz

        # scale by model velocities (broadcast)
        jmx *= xdotx[:, None]
        jmy *= xdoty[:, None]
        jmz *= xdotz[:, None]

        # assemble jm (M,5)
        jm = torch.empty((M,5), device=device)
        jm[:,0] = jmx[:,0] + jmy[:,0]            # jxi
        jm[:,1] = jmz[:,0]                       # jxo
        jm[:,2] = jmx[:,1] + jmy[:,1]            # jyi
        jm[:,3] = jmz[:,1]                       # jyo
        jm[:,4] = jmx[:,2] + jmy[:,2] + jmz[:,2] # jz

        return jm

    # ------------------------------------ End of Simulation Prop Functions --------------------------------------- #
