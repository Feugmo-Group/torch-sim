import os
from os import PathLike
import sys
from typing import Sequence, Optional, Tuple, Union, Mapping, Callable, Literal
import ase
import numpy as np
from math import comb
import torch
import torch_sim as ts
from torch_sim import fire_init, fire_step, gradient_descent_init, gradient_descent_step
from torch_sim.state import SimState
from torch_sim.integrators import MDState
from torch_sim.models.interface import ModelInterface
from torch_sim.autobatching import calculate_memory_scalers, estimate_max_memory_scaler

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
# ----------------------------------------------------------------------

# --------------------------------------------------------------------
# Pending: (minor) Documentation and Better Comments + Logging info and variable name changes

# HNEMDEC Method for determining Heat Transport Coefficients
# Homogenous Non-Equilibrium Molecular Dynamics method based Evans-Cummings algorithm
# Citation: ?
# --------------------------------------------------------------------

import logging
from logging.handlers import RotatingFileHandler
logger = logging.getLogger(__name__)

class HNEMDEC(ImplementationBase, DataSetIO):
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
        super().__init__(method='HNEMDEC', logger=logger, system_state=system_state, temp_K=temp_K, timestep_ps=timestep_ps,
                         nsteps_total=nsteps_total, model=model, n_simulations=n_simulations,
                         model_memory_scaling=model_memory_scaling, perform_equilibration=perform_equilibration,
                         n_equilibration_steps=n_equilibration_steps, store_equilibration_data=store_equilibration_data,
                         equilibration_quantities_functions=equilibration_quantities_functions,
                         perform_geometry_optimization=perform_geometry_optimization, geometry_optimizer=geometry_optimizer,
                         optimizer_convergence_fn=optimizer_convergence_fn, data_folder_path_abs=data_folder_path_abs,
                         simulation_data_filename=simulation_data_filename,
                         equilibration_data_filename=equilibration_data_filename, display_pbar=display_pbar,
                         log_to_file=log_to_file)

    # ------------------------------------------------------------------------------------------------------------- #
    #                                      Pre-Simulation Functions:
    # ------------------------------------------------------------------------------------------------------------- #
    def run_simulation(
            self,
            n_components: int,
            Fe_vec: torch.Tensor,
            sample_interval: int,
            window_size: int,
    ) -> None:
        # ---
        self.Fe_vec = Fe_vec
        self.sample_interval = sample_interval
        self.window_size = window_size
        sys_components = torch.unique(self.system_state.atomic_numbers).shape[0]
        if  sys_components != n_components:
            print(f"Probably Raise an error. Hey, dude your system has more than {n_components} atoms, using torch.unique(system_state.atomic_numbers).shape[0] i got this value: {sys_components}")
        self.n_components = n_components
        # Determine the component mask:
        self.comp_lists = torch.unique(self.system_state.atomic_numbers).tolist()
        state_mask = []
        for system_idx in torch.unique(self.system_state.system_idx).tolist():
            system_mask = []
            system_atoms  = self.system_state.atomic_numbers[self.system_state.system_idx == system_idx]
            for components in range(self.n_components):
                mask = system_atoms == self.comp_lists[components]
                system_mask.append(system_atoms[mask])
            state_mask.append(system_mask)

        # Perform the geometry optimization:
        if self.perform_geometry_optimization:
            self.system_state = self._perform_geometry_optimization(self, logger)

        # Perform the equilibration steps
        if self.perform_equilibration:
            self.system_state = self._perform_equilibration(self, logger)

        self.corr_calc = [None] * self.n_simulations
        corr_props = {}
        for component in range(self.n_components):
            corr_props[f'HAC_{component}'] = self._compute_heat_flux
        for i in range(self.n_simulations):
            corr_calc = CorrelationCalculator(
                window_size=window_size,
                properties=corr_props,
                device=self.device,
                normalize=True,
            )
            self.corr_calc[i] = corr_calc

        self.corr_calc_count, self.n_comp_count = 0, 0
        self.ntriu_entries = int((self.n_components * (self.n_components + 1)) / 2) # including diagonal entries
        # Setting the property recorder to only record crucial data during the simulation
        simulation_prop_to_record = {
            sample_interval: {
                "gamma_triu_entries": self._retrieve_gamma_triu_entries, # unscaled (just the cross correlation value)
            },
        }

        simulation_file_reporter =  ts.TrajectoryReporter(
            filenames=self.simulation_parameters['simulation_filepath'],
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
            pbar={'ascii': True, "desc": 'Simulation: ', 'colour': 'magenta'} if self.display_pbar else None,
        )
        logger.info("Simulation Completed.")

    def post_processing(self):
        for simulation_index in range(len(self.n_simulations)):
            simulation_file = self.simulation_filepath[simulation_index]
            # Load the gamma entries from the simulation file (first value is a dummy value)
            # ((nsteps_total // sample_interval), k)
            gamma_entries_unscaled = torch.tensor(self.load_property(simulation_file, 'gamma_triu_entries')[1:,:], device=self.device, dtype=self.dtype)


            # Compute the Green-Kubo Scaled Gamma Entries
            vol = torch.det(self.system_state.cell[simulation_index, :,:])
            factor = 1/(vol*kB)
            gamma_entries = factor * torch.cumulative_trapezoid(gamma_entries_unscaled, dim=1, dx=(self.simulation_timestep*self.sample_interval))

            # Create the matrix from triu entries
            gamma_matrix = self.batch_upper_triangle_to_symmetric(gamma_entries)

            # Calculating the thermal conductivity:
            inverse_matrix = torch.inverse(gamma_matrix)
            gamma_value_md = inverse_matrix[:,0,0] # MD_units
            gamma_value_si = gamma_value_md * (eV_to_J / (Angs_to_m * ps_to_s)) # SI_units
            thermal_conductivity = 1/(self.temp_K ** 2 * gamma_value_si)
            self.store_property(filepath=simulation_file,property_name='thermal_conductivity',property_dataset=thermal_conductivity, steps=self.nsteps_total)
            logger.info(f"""Simulation {simulation_index} Results: K_μ: {torch.mean(thermal_conductivity)} (W/mK), K_σ: {torch.std(thermal_conductivity)} (W/mK)""")

        return None

    # ------------------------------------------------------------------------------------------------------------- #
    #                    Functions used to calculate simulation property during integration steps:
    # ------------------------------------------------------------------------------------------------------------- #
    # -------------------
    # Steps to compute/store the RTC during HNEMD approach
    # -------------------

    def _retrieve_gamma_triu_entries(self, state, *args):
        # Calculate the correlation using the ith calc (if n_components=0)
        calculator = self.corr_calc[self.corr_calc_count]
        calculator.update(state)
        ac_dict = calculator.get_auto_correlations()
        cc_dict = calculator.get_cross_correlations()
        matrix_triu_entries, cc_count = torch.zeros(size=(self.ntriu_entries, 1), device=self.device, dtype=self.dtype), 0

        # Row-major upper-triangle order (including diagonal): for each name1, its
        # diagonal (self) entry from ac_dict followed by its off-diagonal entries
        # (name1, name2) with name2 later in property order, from cc_dict.
        names = list(calculator.properties)
        for i, name1 in enumerate(names):
            if name1 in ac_dict:
                matrix_triu_entries[cc_count] = ac_dict[name1][-1]
                cc_count += 1
            for name2 in names[i + 1:]:
                key = (name1, name2)
                if key in cc_dict:
                    matrix_triu_entries[cc_count] = cc_dict[key][-1]
                    cc_count += 1


        # Update the calc counter (system_state wise)
        if self.corr_calc_count+1 == self.n_simulations:
            self.corr_calc_count = 0
        else:
            self.corr_calc_count += 1

        return matrix_triu_entries

    def _compute_heat_flux(self, state: MDState):
        # Replace this by using the new state_mask and n_corr_calc count
        mask = (state.atomic_numbers == self.comp_lists[self.n_comp_count])
        out = self.model(state)
        momenta = state.momenta[mask]           # shape: (n_atoms, 3)
        masses = state.masses[mask]             # shape: (n_atoms,)
        virials = out['virials'][mask]          # shape: (n_atoms, 3, 3)
        energy  = out['energies'][mask]
        vol = torch.linalg.det(state.cell)
        # Calculate the Jq_k term
        # Jq_k = sum(pi/mi * Ei)  where Ei = pi^2/mi + U_i
        # t_1 = pi* |pi|^2/2m^2 | t_2 = pi/mi * U_i
        momenta_norm = torch.norm(momenta, dim=1)
        momenta_squared = momenta_norm * momenta_norm
        denominator = 2 * torch.pow(masses,2)
        t1 = momenta * momenta_squared.unsqueeze(1) / denominator.unsqueeze(1)
        t2 = energy.unsqueeze(-1) * (momenta / masses.unsqueeze(-1))
        Jq_kinetic = torch.sum(t1+t2, dim=0)

        # Calculate the Jq_p term
        per_atom_heat_current_from_potential = torch.einsum('bmn,bn -> bm', virials, momenta/masses.unsqueeze(1)) # (n_atoms, 3)
        Jq_potential = torch.sum(per_atom_heat_current_from_potential, dim=0) # (3,)

        # Update the comp_count:
        if self.n_comp_count+1 == self.n_components:
            self.n_comp_count = 0
        else:
            self.n_comp_count += 1

        # Return the heat_flux (MD units)
        return (1/vol)*(Jq_kinetic + Jq_potential)
    # ------------------------------------ End of Simulation Prop Functions --------------------------------------- #
    @staticmethod
    def batch_upper_triangle_to_symmetric(batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert batch of upper triangle entries to symmetric matrices.

        Args:
            batch_tensor: (N, k) tensor where k are upper triangle entries
                          (including diagonal, row-major order)

        Returns:
            (N, n, n) symmetric matrices where n is determined from k
        """
        N, k = batch_tensor.shape

        # Determine n from k: k = n(n+1)/2
        # (k = # of entries in the triu, n = # of dimensions of the square matrix/no of components)
        # Solving for n: n = (-1 + sqrt(1 + 8k)) / 2
        n = int((-1 + np.sqrt(1 + 8*k)) / 2)

        # Get upper triangle indices
        triu_indices = torch.triu_indices(n, n)
        row_idx, col_idx = triu_indices

        # Create output tensor
        matrices = torch.zeros(N, n, n, dtype=batch_tensor.dtype, device=batch_tensor.device)

        # Fill upper and lower triangle for all batches at once (using symmetry)
        matrices[:, row_idx, col_idx] = batch_tensor
        matrices[:, col_idx, row_idx] = batch_tensor

        return matrices

    @staticmethod
    def calculate_external_force(
            energies: torch.Tensor,
            momenta: torch.Tensor,
            masses: torch.Tensor,
            virials: torch.Tensor,
            Fe_vec: torch.Tensor
    ) -> torch.Tensor:
        "F_ext or driving force for HNEMDEC | Not sure if external_force - torch.mean(external_force) is required"
        n_atoms, m_tot = masses.shape[0], torch.sum(masses)
        kT=ts.calc_kT(masses=masses, momenta=momenta)
        identity_matrix = torch.eye(3)
        t1 = torch.einsum('n,ij -> nij', energies, identity_matrix) + virials
        t2 = torch.einsum('n,ij-> nij', (masses/m_tot), torch.sum(t1,dim=0))
        t3_inter = kT * (m_tot - n_atoms*masses)/(n_atoms*m_tot)
        t3 = torch.einsum('n,ij-> nij', t3_inter, identity_matrix) + virials
        F_inter = (t1 - t2 -t3)
        return torch.einsum('bmn,n-> bm', F_inter, Fe_vec)