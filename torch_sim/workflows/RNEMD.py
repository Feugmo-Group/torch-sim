import os
import sys
from os import PathLike
from typing import Sequence, Optional, Tuple, Union, Mapping, Callable
import h5py
import numpy as np
import ase
import torch
from torch_sim.state import SimState
from torch_sim.integrators import MDState
from torch_sim.models.interface import ModelInterface
from torch_sim import fire_init, fire_step
import torch_sim as ts

# --------------------------------------------------------------------
# Reverse Non-Equilibrium Molecular Dynamics (RNEMD) based on Florian Muller-Plathe's Paper
# Citation:
# Florian Müller-Plathe; A simple nonequilibrium molecular dynamics method for calculating the thermal conductivity.
# J. Chem. Phys. 8 April 1997; 106 (14): 6082–6085. https://doi.org/10.1063/1.473271
# --------------------------------------------------------------------

# ----------------------------------------------------------------------
#            Physical constants (SI) & Unit Conversions Value
# ----------------------------------------------------------------------
kB          = 1.380_649e-23                     # J·K⁻¹
ps_to_s     = 1e-12                             # Convertion value: picoseconds to seconds
eV_to_J     = 1.6021766208e-19                  # Convertion value: eV to Joules
Angs_to_m   = 1e-10                             # Convertion value: Angstrom to meter
# ----------------------------------------------------------------------

import logging
from logging.handlers import RotatingFileHandler
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------- #
#                              Velocity Exchange Related Functions:
# ------------------------------------------------------------------------------------------------------------- #

def perform_velocity_exchange_step(system_state: MDState, n_exchanges: int,
                                   lower: torch.Tensor, upper: torch.Tensor, filepath: PathLike)-> MDState:
    """
    Parameters:
        system_state (MDState): system state object to modify
        n_exchanges (int): Number of atoms to swap
        lower (torch.Tensor): Tensor defining lower bound for each slab
        upper (torch.Tensor): Tensor defining upper bound for each slab
        filepath (PathLike): Path where to save the v_exchange information (intmd_file)

    Returns:
        system_state (MDState): modified system state

    Note: Instead of defining the system as in the paper, we define:
          hot slab at slab_0 and cold slab at slab_N

    Steps For Velocity Exchange Step:
        1. Find 'n_exchanges' number of coldest atom in slab_1
        2. Find 'n_exchanges' number of hottest atom in slab_N
        ----------- For each exchange -----------
        3. Find the particles in system_state
        4. Update the system_state by Swapping particle velocity and momenta around
        Note: # We don't modify the Positions, Forces, Masses, (Potential) Energy
        -----------------------------------------
        5. Return the updated system_state
    """
    # Calculate slab_idx, slabwise_velocity
    device = system_state.device
    slabwise_velocities, slabwise_masses, slab_idx = classify_particle_slab(system_state, lower.to(device), upper.to(device))

    # Slabwise_velocities and slab_idx can be calculated from system_state, lower and upper
    # Step 1: Find 'n_exchanges' coldest atom(s) in Slab_0
    slab_0_velocities = slabwise_velocities[0]
    slab_0_speed = torch.linalg.norm(slab_0_velocities, dim=1, ord = 2)
    negative_slab_0_speed = -1 * slab_0_speed
    min_values, min_values_idx = torch.topk(negative_slab_0_speed, n_exchanges, dim=-1)
    min_values, min_values_idx = torch.abs(min_values).tolist(), min_values_idx.tolist() # Convert abs(tensor) to list

    # Step 2: Find 'n_exchanges' hottest atom in Slab_N
    slab_n = slabwise_velocities[-1]
    slab_n_speed = torch.linalg.norm(slab_n, dim=1, ord = 2)
    max_values, max_values_idx = torch.topk(slab_n_speed, n_exchanges, dim=-1)
    max_values, max_values_idx = max_values.tolist(), max_values_idx.tolist() # Convert tensor to list

    # Generate the v_hot_values and v_cold_values list:
    v_hot_values: list = max_values                                  # Note: this is |v| and not \vec{v}
    v_cold_values: list = min_values                                 # Note: this is |v| and not \vec{v}

    # Save the v_hot_values and v_cold_values to the "intmd_vexchange_data.h5" inside data_folder_path_abs:
    RNEMD.append_to_intmd_file(v_hot_values, v_cold_values, filepath=filepath)

    for ith_exchange in range(n_exchanges):
        min_value_idx = min_values_idx[ith_exchange]
        max_value_idx = max_values_idx[ith_exchange]

        # Step 3: Finding the particle in the system_state
        idx_coldest_atom = find_particle(slab_idx, 0, min_value_idx)
        idx_hottest_atom = find_particle(slab_idx, (len(slabwise_velocities)-1), max_value_idx)

        # Step 4: Swap the velocity and momenta to Update the system_state (for coldest_atom and hottest_atom)
        # Swapping Velocities:
        hot_slab_particle_velocity, cold_slab_particle_velocity = system_state.velocities[idx_hottest_atom,:].detach(), system_state.velocities[idx_coldest_atom,:].detach()
        system_state.velocities[idx_coldest_atom,:] = hot_slab_particle_velocity
        system_state.velocities[idx_hottest_atom,:] = cold_slab_particle_velocity

        # Swapping Momenta:
        hot_slab_particle_momenta, cold_slab_particle_momenta = system_state.momenta[idx_hottest_atom,:].detach(), system_state.momenta[idx_coldest_atom,:].detach()
        system_state.momenta[idx_coldest_atom,:] = hot_slab_particle_momenta
        system_state.momenta[idx_hottest_atom,:] = cold_slab_particle_momenta

    # Step 5: Return the updated system_state
    return system_state


def find_particle(slab_idx: torch.Tensor, slab_no: int,particle_idx_in_slab: int):
    """
    Description: Find the system_state index of a particular particle given its index in the jth slab
    Parameters:
        slab_idx: Tensor containing slabwise classification of system_state particles
        slab_no: Slab number in which the particle is found in ∈ (0, n_slab-1)
        particle_idx_in_slab: index of the particle in slab_no specified before
    Returns:
        particle_system_state_idx
    """
    # Get all the indices of particles that are in slab: "slab_no"
    particle_indices = (slab_idx == slab_no).nonzero(as_tuple=True)[0]

    # Find the system_state index of the ith particle in the slab_no
    particle_system_state_idx = particle_indices[particle_idx_in_slab - 1].item()

    return particle_system_state_idx

# ------------------------------------ End of v_exchange functions ----------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------- #
#                              Particle Classification Related Functions:
# ------------------------------------------------------------------------------------------------------------- #

def classify_particle_slab(system_state, lower, upper) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """
    Description: Classify particles in a system into their respective slabs based on upper and lower bound of slabs
                 and returns slab-wise particle velocities
    Returns:
    (N = n_slabs)
    slabwise_velocities : List of N Tensors where ith entry contains all the particle velocities in the ith slab
    slabwise_masses     : List of N Tensors where ith entry contains all the masses in the ith slab
    slab_idx            : Tensor of shape (N,) containing particle classification information
                           ith element of the slab_idx relays the slab no. in which the ith particle is classified under.

    Could be modified to return:
    slabwise_positions  : List of N Tensors where ith entry contains all the particle positions in the ith slab
    slabwise_momenta    : List of N Tensors where ith entry contains all the particle moments in the ith slab
    """
    masses = system_state.masses                # (ndim,)
    positions = system_state.positions          # (ndim, 3)
    velocities = system_state.velocities        # (ndim, 3)

    # Determining the No. of Slabs based on the dimensions of lower
    if lower.shape[0] == upper.shape[0]:
        n_slabs = lower.shape[0]
    else:
        logger.error("Shape for Lower and Upper Tensor weren't the same: {} != {}".format(lower.shape, upper.shape))
    z_coordinate = positions[:, 2:3] # Slicing the positions tensor

    # Vectorised range check (n_slabs, ndim, 1)
    range_match = (z_coordinate.squeeze(1) >= lower) & \
                  (z_coordinate.squeeze(1) <= upper)

    # First matching slab index (or -1 if no matches)
    slab_idx = torch.argmax(range_match.float(), dim=0)   # (ndim,)
    no_match = ~range_match.any(dim=0)                    # (ndim,) bool
    slab_idx[no_match] = -1

    # ----------------------- Error Handling ----------------------- #
    # Attempt error diagnosis (if the classification fails) and try to find the closest place where it could still be within the simulation box
    if torch.any(no_match).item():
        error_fixed, slab_idx = perform_error_diagnosis(positions, z_coordinate, lower, upper, slab_idx)
        if not error_fixed:
            # Provide detailed error context
            problematic_indices = torch.where(slab_idx[no_match])[0]
            error_details = {
                "Total unclassified particles": len(problematic_indices),
                "Problematic particle indices": problematic_indices.tolist(),
                "Z-coordinates": z_coordinate[no_match].tolist(),
                "Slab boundaries": {
                    "lower": lower.tolist(),
                    "upper": upper.tolist()
                }
            }
            # Raise the more informative error:
            logger.error(f"Unable to classify all particles.\nError Details:\n{error_details}")
    # --------------------------------------------------------------- #
    slabwise_velocities = []
    slabwise_masses = []
    for i in range(n_slabs):
        # Create an "Empty" mask that has the same "dimensions" as positions
        ith_slab_mask = torch.zeros(positions.size(0), dtype=torch.bool)
        # Modify the mask to display which particles are in the ith Slab i.e. torch.tensor([True, False, True, ...])
        # indicates 0th, 2nd and so on  particles are in the ith slab
        ith_slab_mask[slab_idx == i] = True  # shape: (ndim, ), bool

        # Append the new tensor containing corresponding property of particles in the ith slab
        slabwise_velocities.append(velocities[ith_slab_mask])
        slabwise_masses.append(masses[ith_slab_mask])

    return slabwise_velocities, slabwise_masses, slab_idx

def perform_error_diagnosis(positions, z_coordinate, lower, upper, slab_idx) -> tuple[bool, torch.Tensor]:
    """
    Diagnose and attempt to resolve slab classification errors.

    Parameters:
        positions (torch.Tensor): Full position tensor
        z_coordinate (torch.Tensor): Z-coordinates of particles
        lower (torch.Tensor): Lower bounds of slabs
        upper (torch.Tensor): Upper bounds of slabs
        slab_idx (torch.Tensor): Current slab indices

    Returns:
        bool: Whether all classification errors were successfully resolved
        new_slab_idx (torch.Tensor): Updated slabwise indices
    """
    # Convert to lists for easier manipulation
    zcoords = z_coordinate.squeeze(1).tolist()
    # Collect all unique interval bounds
    lower_list = lower.flatten().tolist()
    upper_list = upper.flatten().tolist()
    range_list = sorted(set(lower_list + upper_list))
    # Track whether we've made any corrections
    corrections_made = False

    for i, (z_coord, pos) in enumerate(zip(zcoords, positions)):
        if slab_idx[i] == -1:
            # Find the closest interval boundary
            closest_interval_value = min(range_list, key=lambda x: abs(x - z_coord))
            furthest_interval_value = max(range_list, key=lambda x: abs(x - z_coord))
            try:
                # Attempt to correct the z-coordinate
                z_coordinate[i] = torch.tensor(furthest_interval_value + (z_coord - closest_interval_value),
                                               dtype=z_coordinate.dtype,
                                               device=z_coordinate.device)
                # Re-check classification after correction
                range_match = (z_coordinate.squeeze(1) >= lower) & \
                              (z_coordinate.squeeze(1) <= upper)
                # Update slab indices
                new_slab_idx = torch.argmax(range_match.float(), dim=0)
                no_match = ~range_match.any(dim=0)
                new_slab_idx[no_match] = -1
                # Update slab_idx if correction successful
                if new_slab_idx[i] != -1:
                    slab_idx[i] = new_slab_idx[i]
                    corrections_made = True
            except Exception as e:
                logger.error(f"Exception:  Unexpected error during correction: {e}")

    # Final classification check
    final_range_match = (z_coordinate.squeeze(1) >= lower) & \
                        (z_coordinate.squeeze(1) <= upper)
    final_no_match = ~final_range_match.any(dim=0)

    # Return True if no unresolved misclassifications remain
    # we also return slab_idx so that our error_details updated information about the problematic coords
    return not torch.any(final_no_match).item(), slab_idx

# --------------------------------- End of Classify Particles Functions ------------------------------------------- #

class RNEMD():
    """
    Driver for reverse non-equilibrium molecular dynamics (RNEMD) simulations.
    RNEMD(system_state=None, nslabs=20, n_atoms=None, atomic_number=None, box_dimensions_angs=None, device=None, dtype=None, pbc=None)

    Parameters:
        system_state (SimState | None): Pre-built SimState representing a single-system, single-element configuration. If provided, it is used directly. Must contain diagonal cell.
        nslabs (int): Number of slabs to partition the simulation box into along the transport direction.
    """
    def __init__(self, system_state: SimState, nslabs: int=20) -> None:
        """
        Initialize RNEMD object from an existing SimState
        To initialize it based on a set of construction parameters (nslabs, n_atoms, atomic_number, box_dimensions_angs, device, dtype & pbc) use RNEMD.create_simple_system().

        Raises
            NotImplementedError: if SimState contains multiple systems or multiple atomic species.
            ValueError:  if slabs isn't > zero

        Side effects
            Sets attributes: system_state, n_atoms, atomic_number, atomic_mass_amu, simulation_device, dtype, nslabs, box dimensions (x,y,z).
            Calls _generate_system_slabs() to partition the box.
        """
        # Check for single system
        if system_state.n_systems > 1:
            raise NotImplementedError("For the current implementation: SimState must contain a single system. Multiple systems detected.")
        # Check for system with multiple elements
        if torch.unique(system_state.atomic_numbers).shape[0] > 1:
            raise NotImplementedError("For the current implementation: SimState must contain a single type of atom. Multiple atomic species detected")
        # Ensure that the nSlab is even:
        if nslabs <= 0:
            raise ValueError("Number of slabs must be greater than zero.")

        # Initialize the simulation variables
        self.system_state = system_state
        self.n_atoms = system_state.n_atoms
        self.atomic_number = system_state.atomic_numbers[0].item()
        self.atomic_mass_amu = ase.data.atomic_masses[self.atomic_number]
        if not torch.all(system_state.cell * (1 - torch.eye(system_state.cell.size(0), device=system_state.cell.device)) == 0):
            # Cell isn't diagonalized (the system_state has not conventional setup for the simulation_box)
            raise ValueError("SimState.cell must be a diagonal matrix")
        self.x, self.y, self.z = system_state.cell[0,0,0].item(), system_state.cell[0,1,1].item(), system_state.cell[0, 2,2].item()
        self.simulation_device = system_state.device
        self.dtype = system_state.dtype
        self.nslabs = nslabs
        self.system_state = system_state

        # Generate the slab classification
        self._generate_system_slabs()

    @classmethod
    def create_simple_system(cls, atomic_number:int, box_dimensions_angs:tuple[float, float, float], n_atoms: int,
                             pbc:list[bool]=[False, False, True], device=torch.device('cuda'), dtype=torch.float32,
                             ) -> SimState:
        """
        Create a simple atoms based system and return a corresponding SimState object for RNEMD Simulations.
        Parameters:
            atomic_number (int | None): Atomic number for all atoms when constructing from parameters.
            box_dimensions_angs (tuple[float, float, float] | None): Box dimensions in Å when constructing from parameters (Lx, Ly, Lz).
            n_atoms (int | None): Number of atoms to create when constructing a new SimState from parameters.
            pbc (list[bool] | None): Periodic boundary conditions for the constructed system.
            device: Target compute device for tensors (e.g., torch.device).
            dtype: Torch dtype for simulation tensors (e.g., torch.float32).
        Returns:
            SimState object
        """
        x, y, z = box_dimensions_angs
        # Generating Uniformly Distributed Coordinate to "scatter" the atoms in the SimulationBox
        rng = np.random.default_rng()
        system_coords = rng.uniform(low=[0, 0, 0],high=[x, y, z], size=(n_atoms, 3))

        # Initialize the Atoms Object
        system_atoms = ase.Atoms(numbers=[atomic_number] * n_atoms, positions=system_coords, cell=[x, y, z], pbc=pbc)

        # Initialize the TorchSim SimState
        system_simstate = ts.state.initialize_state(system_atoms, device, dtype)

        return system_simstate

    # ------------------------------------------------------------------------------------------------------------- #
    #                                      Pre-Simulation Functions:
    # ------------------------------------------------------------------------------------------------------------- #
    @staticmethod
    def _perform_geometry_optimization(system_state, model, opt_func =None):
        # Optimize/Relax the geometry of the randomly populated SimulationBox:
        # i.e. minimize the potential energy and the forces in the system
        def custom_convergence_fn(state, last_energy):
            energy_convergence_fn = ts.runners.generate_energy_convergence_fn(energy_tol=1e-6)
            force_convergence_fn = ts.runners.generate_force_convergence_fn(force_tol=1e-6, include_cell_forces=False)
            energy_output = energy_convergence_fn(state, last_energy)
            force_output = force_convergence_fn(state, last_energy)
            return torch.logical_and(energy_output, force_output)

        # energy_convergence_fn = ts.runners.generate_energy_convergence_fn(energy_tol=1e-6)
        relaxed_state = ts.runners.optimize(
            system=system_state,
            model=model,
            optimizer= (fire_init, fire_step),
            convergence_fn= custom_convergence_fn if opt_func is None else opt_func,
        )
        logger.info(f"[Geometry optimization finished] Post-Relaxation Energy is {model(relaxed_state)['energy'].item():.2f} eV")
        return relaxed_state

    @staticmethod
    def _perform_equilibration(system_state: SimState, model: ModelInterface, simulation_parameters: dict) -> MDState:
        """
        Run equilibration integration on the current system_state using the stored simulation parameters.

        Behavior:
            Reads equilibration-related parameters from self.simulation_parameters.
            Registers property calculators with a TrajectoryReporter if store_equil_data is enabled.
            Updates self.system_state with final MDState returned by the integrator.
        """

        n_equil_step = simulation_parameters['n_equil_step']
        store_equil_data = simulation_parameters['store_equil_data']
        equi_prop_to_record = {
            1: {
                "system_temperature": RNEMD._calculate_system_temperature,
                "system_potential": RNEMD._calculate_system_potential
            },
        }
        if store_equil_data:
            equilibrium_file_reporter = ts.TrajectoryReporter(
                filenames=simulation_parameters['equilibration_filepath'],
                prop_calculators=equi_prop_to_record,
            )
            logger.info(f"Equilibration Data filepath set to {simulation_parameters['equilibration_filepath']}")
        else:
            equilibrium_file_reporter = None

        system_MDState: MDState = ts.integrate(
            system = system_state,
            model  = model,
            integrator = ts.Integrator.nvt_nose_hoover,
            n_steps = n_equil_step,
            timestep = simulation_parameters['timestep_ps'],
            temperature = simulation_parameters['temp_K'],
            trajectory_reporter = equilibrium_file_reporter,
        )
        sys_temp_k = ts.quantities.calc_temperature(masses=system_MDState.masses,
                                                    momenta=system_MDState.momenta,
                                                    )
        logger.info(f"[Equilibration Completed] Post-Equilibration System Temperature: {sys_temp_k:.2f} K")
        if torch.isnan(system_MDState.positions).any():
            logger.debug(f"NaN values in system_MDState.positions after equilibration")
        return system_MDState

    # ------------------------------------ End of Pre-Simulation Functions --------------------------------------- #

    def run_simulation(
            self,
            model: ModelInterface,
            timestep_ps: float,
            W: int,
            nsteps_total: int,
            temp_K: float,
            perform_equilibration: bool=True,
            perform_geometry_optimization: bool=True,
            optimization_function: Callable=None,
            n_equil_step: int=10_000,
            store_equil_data: bool =False,
            n_exchanges_per_step: int = 1,
            data_folder_path_abs: PathLike=None,
            equilibration_data_filename: str ='equilibrium_data',
            simulation_data_filename: str='simulation_data',
            log_to_file: bool=True,
    ) -> None:
        """
        Configure and execute the main RNEMD run.

        Parameters:
            model (ModelInterface): The interatomic model to use with ts.integrate.
            timestep_ps (float): Integration timestep in picoseconds.
            W (int): Frequency (in integration steps) at which velocity exchanges or system modifiers are applied.
            nsteps_total (int): Total number of integration steps to run.
            temp_K (float): Desired Simulation Temperature in Kelvin
            perform_equilibration (bool): If True, run equilibration before simulation.
            perform_geometry_optimization (bool): If True, run geometry optimization before simulation (recommended).
            optimization_function (Callable): function to optimize using (optional, defaults to optimize both force and energy)
            n_equil_step (int): Number of equilibration steps.
            store_equil_data (bool): Write equilibration properties to disk when True.
            n_exchanges_per_step (int): Number of pair exchanges to perform every W steps.
            data_folder_path_abs (PathLike): Directory to store output files.
                Although the code works with relative paths, specifying absolute path is recommended
                If None, defaults to 'RNEMD_simulation_data' folder in caller's directory (cwd) (not recommended).
            equilibration_data_filename (str): Filename for equilibration_data file
            simulation_data_filename (str): Filename for simulation_data file
            log_to_file (bool): Whether to save the logger to log

        Logs the following error using Logger:
            ValueError(): if nsteps_total is not divisible by W.

        Side effects:
            Sets self.simulation_parameters and self.system_modifiers.
            May modify n_exchanges_per_step to obey internal limits (emit a warning rather than raise).
            Writes files to data_folder_path_abs (in order to save the simulation and equilibration data).
        """
        self.model = model
        self.temp_K = temp_K
        self.timestep_ps = timestep_ps
        self.nsteps_total = nsteps_total
        # Check if nstep_total is divisible by W:
        if not nsteps_total % W ==0:
            logger.error("ValueError: nsteps_total must be divisible by W")
        else:
            self.total_exchange_steps = nsteps_total//W
        # Override n_exchanges value if beyond a certain value
        if n_exchanges_per_step > int(self.n_atoms/ (self.nslabs * 5)):
            initial_value = n_exchanges_per_step
            n_exchanges_per_step = int(self.n_atoms/ (self.nslabs * 5)) # arbitrary max_value
            logger.debug(f"Overriding n_exchanges to {n_exchanges_per_step} instead of the specified {initial_value}")
        # Initialise self.data_folder_path_abs and create the path if it doesn't exist
        if data_folder_path_abs is None:
            directory = os.getcwd() # Get the current working directory (from which RNEMD is being called)
            self.data_folder_path_abs = os.path.join(directory, 'RNEMD_simulation_data')
        else:
            os.makedirs(data_folder_path_abs, exist_ok=True)
        os.makedirs(self.data_folder_path_abs, exist_ok=True)

        if log_to_file:
            self._configure_logger(logfile=os.path.join(self.data_folder_path_abs, 'simulation_logs.log'))
            logger.info(f"Logging to {os.path.join(self.data_folder_path_abs, 'simulation_logs.log')}")
        else:
            logger.info("Logger initialised to not log to a file")
        simulation_parameters = {'model': model, 'timestep_ps': timestep_ps, "W": W, "nsteps_total": nsteps_total, 'temp_K': temp_K,
                                 "perform_equilibration": perform_equilibration, "n_equil_step": n_equil_step,
                                 'store_equil_data': store_equil_data, 'n_exchanges_per_step': n_exchanges_per_step,
                                 'equilibration_filepath': os.path.join(self.data_folder_path_abs, f"{equilibration_data_filename}.h5"),
                                 'simulation_filepath': os.path.join(self.data_folder_path_abs, f"{simulation_data_filename}.h5"),
                                 'intmd_exchange_filepath': os.path.join(self.data_folder_path_abs, "intmd_vexchange_data.h5"),
                                 'logger_filepath': os.path.join(self.data_folder_path_abs, f"simulation_logs.log"),
                                 }

        self.simulation_parameters = simulation_parameters

        # Perform the geometry optimization:
        if perform_geometry_optimization:
            logger.info("Performing Geometry Optimization...")
            self.system_state = self._perform_geometry_optimization(self.system_state, model, optimization_function)

        # Perform the equilibration steps
        if perform_equilibration:
            logger.info("Performing Equilibration...")
            system_state = self._perform_equilibration(self.system_state, model, simulation_parameters)

        # Initialising the intmd_file: n_store_calls = self.total_exchange_steps, n_items_per_list = n_exchanges_per_step
        self.initialise_intmd_file(simulation_parameters['intmd_exchange_filepath'], int(self.total_exchange_steps), int(n_exchanges_per_step))

        system_modifiers = {'modification_freq': W, "modification_func": perform_velocity_exchange_step,
                            "kwargs": {'n_exchanges': n_exchanges_per_step, 'lower': self.lower, 'upper': self.upper,
                                       'filepath': self.simulation_parameters['intmd_exchange_filepath']}
                            }
        self.system_modifiers = system_modifiers

        # Setting the property recorder to only record crucial data during the simulation
        # (since we can calculate thermal conductivity and other quantities based on slabwise_temperature and v_exchange_info)
        simulation_prop_to_record = {
            W: {
                "system_temperature": self._calculate_system_temperature,
                "system_potential": self._calculate_system_potential,
                "slabwise_temperature": self._calculate_slabwise_temperature,
            },
        }
        simulation_reporter = ts.TrajectoryReporter(
            filenames=self.simulation_parameters['simulation_filepath'],
            prop_calculators=simulation_prop_to_record,
        )
        logger.info(f"Simulation Data filepath set to {self.simulation_parameters['simulation_filepath']}")
        logger.info("Performing simulation...")

        system_state = ts.integrate(
            system = system_state,
            model = model,
            system_modifier=system_modifiers,
            integrator = ts.Integrator.nvt_nose_hoover,
            n_steps = simulation_parameters['nsteps_total'],
            timestep=simulation_parameters['timestep_ps'],
            temperature=temp_K,
            trajectory_reporter=simulation_reporter,
        )
        self.system_state = system_state
        logger.info("Simulation Concluded.")

    def post_simulation_processing(self, compute_running_values:bool=True, save_simulation_results: bool=True) -> None:
        """
        Compute time-averaged and running transport properties from simulation output files and in-memory data.

        Parameters:
            compute_running_values (bool): If True, compute and return/store running thermal conductivity and related running quantities.
            save_simulation_results (bool): If True, write computed properties into the simulation HDF5 file(s).

        Behavior:
            Reads intermediate v-exchange data and simulation trajectory properties.
            Computes slabwise temperature gradients, cumulative v-exchange statistics, and the final thermal conductivity using consistent units.
            Stores computed datasets to disk via 'simulation_file_store_property' when save_simulation_results is True.

        Notes and cautions:
            Expects well-formed files at paths specified in self.simulation_parameters. Validates shapes and step counts.

        """
        logger.info("Performing post-simulation processing...")
        # Calculate important simulation specific quantities
        simulation_exchange_steps = list(range(self.simulation_parameters['W'], self.nsteps_total+1, self.simulation_parameters['W']))
        intmd_file = self.simulation_parameters['intmd_exchange_filepath']
        simulation_file = self.simulation_parameters['simulation_filepath']
        self.simulation_file = simulation_file

        # Get the vexchange data from the intmd file:
        v_hot_list, v_cold_list = self.intmd_file_read_all_data(intmd_file)
        sqred_v_hot_list, sqrd_v_cold_list = torch.pow(torch.tensor(v_hot_list), 2), torch.pow(torch.tensor(v_cold_list), 2)
        vhot_cumulative_list = self._formulate_cumulative_prop(torch.tensor(v_hot_list)) # Generates a running vhot_list at each step
        vcold_cumulative_list = self._formulate_cumulative_prop(torch.tensor(v_cold_list)) # Generates a running vcold_list at each step

        # Get the slabwise_temperature from the simulation_file and then convert it to torch.Tensor
        # slabwise_temperature:  (n_exchange_step, nslabs)
        slabwise_temperature = self.simulation_file_read_property(self.simulation_parameters['simulation_filepath'], 'slabwise_temperature')
        slabwise_temperature = torch.tensor(slabwise_temperature[1:], dtype=self.dtype, device=self.simulation_device, requires_grad=True)

        # Using Slabwise Temperature: Calculate Temperature Gradient at each timestep using torch.gradient()
        # dTdz_over_time:  (n_exchange_steps, nslabs) | ith element corresponds to temperature gradient at ith vexchange step
        dTdz_over_time = torch.abs(torch.gradient(slabwise_temperature, spacing=self.slab_length_m, dim=1)[0]) # testing abs gradient

        # Calculate the Final Time-Averaged (over n_exchange_step) Temperature gradient in z direction
        dTdz_mean_z = torch.mean(dTdz_over_time, dim=1)

        # Reduce dTdz_mean_z to Calculate (scalar) time_averaged temperature gradient
        avg_dTdz = torch.mean(dTdz_mean_z) # < dT/dz >

        # Determine Final Thermal Conductivity
        # Formula used: - numerator/denominator
        # numerator = (0.5 * atomic_mass_amu * (np.sum(vhot_squared) - np.sum(vcold_squared)) * eV_to_J)
        # denominator = (L_x_in_m * L_y_in_m * avg_dTdz * time_elapsed * n_exchanges_per_step)
        L_x_in_m = self.x * Angs_to_m
        L_y_in_m = self.y * Angs_to_m
        total_time_elapsed = self.simulation_parameters['timestep_ps'] * self.simulation_parameters['nsteps_total'] * ps_to_s
        n_exchanges_per_step = self.simulation_parameters['n_exchanges_per_step']
        numerator = 0.5 * self.atomic_mass_amu * (torch.sum(sqred_v_hot_list) - torch.sum(sqrd_v_cold_list)) * eV_to_J
        denominator = L_x_in_m * L_y_in_m * avg_dTdz * total_time_elapsed * n_exchanges_per_step
        final_thermal_conductivity = numerator/denominator

        if compute_running_values:
            logger.info("Calculating running values...")
            #----------------------------------------------------------
            # Initialize and preallocate arrays for running values:
            #----------------------------------------------------------
            # For running energy transfer term: (vexchange)
            running_vhot_squared, running_vcold_squared = [None] * self.total_exchange_steps, [None] * self.total_exchange_steps
            running_sum_vhot_squared, running_sum_vcold_squared = [None] * self.total_exchange_steps, [None] * self.total_exchange_steps

            # For running temperature gradient values:
            running_dTdz_over_time = dTdz_over_time.detach()
            running_dTdz_mean_z, running_avg_dTdz = [None] * self.total_exchange_steps, [None] * self.total_exchange_steps

            # For running thermal conductivity:
            running_thermal_conductivity = [None] * self.total_exchange_steps
            # Note: Not calculating reduced thermal conductivity
            #----------------------------------------------------------


            # Calculate all the running values within a single for loop (computationally intensive [No known optimization available])
            for steps in range(self.total_exchange_steps):
                # Calculate time_elapsed at each vexchange step:
                simulation_step = (steps+1) * self.simulation_parameters['W'] # step+1 since steps starts from zero
                time_elapsed = self.simulation_parameters['timestep_ps'] * simulation_step * ps_to_s

                # Calculate energy transfer (vexchange) term by processing vhot_cumulative_list and vcold_cumulative_list:
                vhot_squared = torch.pow(vhot_cumulative_list[steps], 2)
                vcold_squared = torch.pow(vcold_cumulative_list[steps], 2)

                running_vhot_squared[steps] = vhot_squared
                running_vcold_squared[steps] = vcold_squared

                cumulative_sum_vhot_squared = torch.sum(vhot_squared)
                cumulative_sum_vcold_squared = torch.sum(vcold_squared)

                running_sum_vhot_squared[steps] = cumulative_sum_vhot_squared
                running_sum_vcold_squared[steps] = cumulative_sum_vcold_squared

                # Calculate running temperature_gradient values
                cumulative_dTdz_z = running_dTdz_over_time[:steps]
                running_dTdz_mean_z[steps] = torch.mean(cumulative_dTdz_z, dim=1)
                ith_avg_dTdz = torch.mean(cumulative_dTdz_z)
                running_avg_dTdz[steps] = ith_avg_dTdz.item()

                # Calculate running thermal conductivity using running time averaged temp grad
                # running_thermal_conductivity = - numerator/denominator
                # numerator = (0.5* atomic_mass_amu * (sum(vhot_squared) - sum(vcold_squared)) * eV_to_J)
                # denominator = (L_x_in_m * L_y_in_m * avg_dTdz * time_elapsed * n_exchanges_per_step)
                numerator = 0.5* self.atomic_mass_amu * (cumulative_sum_vhot_squared - cumulative_sum_vcold_squared) * eV_to_J
                denominator = L_x_in_m * L_y_in_m * ith_avg_dTdz * time_elapsed * n_exchanges_per_step
                ith_thermal_conductivity = - numerator / denominator
                running_thermal_conductivity[steps] = ith_thermal_conductivity.item()


        if save_simulation_results:
            logger.info(f"Storing calculated simulation quantities in {simulation_file}...")
            # Store cumulative vexchange info and vexchange info in the simulation file
            self.simulation_file_store_property(simulation_file, 'v_hot_list', v_hot_list, len(simulation_exchange_steps))
            self.simulation_file_store_property(simulation_file, 'v_cold_list', v_cold_list, len(simulation_exchange_steps))
            self.simulation_file_store_property(simulation_file, 'v_hot_cumulative', vhot_cumulative_list, len(simulation_exchange_steps))
            self.simulation_file_store_property(simulation_file, 'v_cold_cumulative', vcold_cumulative_list, len(simulation_exchange_steps))
            # optional: Remove/delete the intmd file
            logger.info(f"Deleting the intermediate file: {intmd_file}")
            os.remove(intmd_file)

            # Store dTdz_over_time, dTdz_mean_z, dTdz in the simulation file
            self.simulation_file_store_property(simulation_file, 'dTdz_over_time', dTdz_over_time, self.simulation_parameters['nsteps_total'])
            self.simulation_file_store_property(simulation_file, 'dTdz_mean_z', dTdz_mean_z, self.simulation_parameters['nsteps_total'])
            self.simulation_file_store_property(simulation_file, 'avg_dTdz', avg_dTdz, self.simulation_parameters['nsteps_total'])

            # Store the final_thermal_conductivity:
            self.simulation_file_store_property(simulation_file, 'final_thermal_conductivity', final_thermal_conductivity, self.simulation_parameters['nsteps_total'])
            if compute_running_values:
                logger.info(f"Storing running values in {simulation_file}")
                # Store running_vhot_squared, cumulative_sum_vhot_squared ... etc.
                self.simulation_file_store_property(simulation_file, 'running_vhot_squared', running_vhot_squared, simulation_exchange_steps)
                self.simulation_file_store_property(simulation_file, 'running_vcold_squared',running_vcold_squared, simulation_exchange_steps)
                self.simulation_file_store_property(simulation_file, 'running_sum_vhot_squared', running_sum_vhot_squared, simulation_exchange_steps)
                self.simulation_file_store_property(simulation_file, 'running_sum_vcold_squared', running_sum_vcold_squared, simulation_exchange_steps)
                # Store running_dTdz_over_time, running_dTdz_mean_z and running_avg_dTdz in simulation file
                self.simulation_file_store_property(simulation_file, 'running_dTdz_over_time', running_dTdz_over_time, simulation_exchange_steps)
                self.simulation_file_store_property(simulation_file, 'running_dTdz_mean_z', running_dTdz_mean_z, simulation_exchange_steps)
                self.simulation_file_store_property(simulation_file, 'running_avg_dTdz', running_avg_dTdz, simulation_exchange_steps)
                # Store running thermal_conductivity
                self.simulation_file_store_property(simulation_file, 'running_thermal_conductivity', running_thermal_conductivity, simulation_exchange_steps)

        logger.info("Post-Simulation Processing Complete!")


    # __________________________________________________________________________________

    # ------------------------------------------------------------------------------------------------------------- #
    #                    Functions used to calculate simulation property during integration steps:
    # ------------------------------------------------------------------------------------------------------------- #
    def _calculate_slabwise_temperature(
            self,
            system_state: MDState,
    )-> torch.Tensor:
        """
        Description: Calculate the slabwise temperature based on the formula given in the paper
        Formula Used : T_k = 1/(3*n_atoms * k_b) * sum(mass_i * v_i^2)
        Returns:
            slabwise_temperature: List of Slabwise temperature

        Citation: Florian Müller-Plathe; A simple nonequilibrium molecular dynamics method for calculating the thermal
        conductivity. J. Chem. Phys. 8 April 1997; 106 (14): 6082–6085. https://doi.org/10.1063/1.473271
        """
        # Calculate slab_idx, slabwise_velocity
        slabwise_velocities, slabwise_masses, slab_idx = classify_particle_slab(system_state, self.lower.to(self.simulation_device), self.upper.to(self.simulation_device))

        slabwise_temperature = []
        for slab in range(len(slabwise_masses)): #
            masses = slabwise_masses[slab]
            n_atoms = masses.shape[0]
            if n_atoms == 0:
                if sum([masses.shape[0] for masses in slabwise_masses]) == self.n_atoms:
                    raise Exception(f"Even though total atoms are conserved, No atom found in slab {slab}. Check your PBC conditions. Unique slabwise classification in the system: {slab_idx.unique()}, Ideally the classification should classify atoms in all of the slabs i.e 0 to {self.n_slabs-1}")
                else:
                    raise Exception(f"Total Atoms aren't being conserved and consequently no atom found in slab {slab}. Check your PBC conditions.")
            velocities = slabwise_velocities[slab]
            # Calculate the normalized velocity squared \vec{v} -> |v| -> |v|^2
            normalized_velocities_squared = torch.pow(torch.linalg.norm(velocities, dim=1, ord = 2), 2)

            # Calculate the kinetic energy term by doting masses and v^2
            kinetic_energy_eV = torch.dot(masses, normalized_velocities_squared) # This is in eV since amu * Ang * fs^-1
            kinetic_energy = kinetic_energy_eV * eV_to_J  #  Converting to Joules
            ith_slab_temperature = ( 1 / (3 * n_atoms * kB) ) * kinetic_energy
            slabwise_temperature.append(ith_slab_temperature.item())
        return torch.tensor(slabwise_temperature, device = self.simulation_device)

    @staticmethod
    def _calculate_system_temperature(state: ts.SimState) -> torch.Tensor:
        return ts.quantities.calc_temperature(masses=state.masses, momenta=state.momenta)

    @staticmethod
    def _calculate_system_potential(state: ts.SimState, model: ModelInterface) -> torch.Tensor:
        return model(state)['energy']

    # ------------------------------------ End of Simulation Prop Functions --------------------------------------- #

    # ------------------------------------------------------------------------------------------------------------- #
    #                    Functions used to store and read data stored in simulation_file:
    # ------------------------------------------------------------------------------------------------------------- #
    @staticmethod
    def simulation_file_read_property(filepath: PathLike, property_name:str):
        """
        Reads the simulation file and returns the property dataset
        Retrieve a named property array from a simulation trajectory file.
        Parameters:
        filepath (PathLike):  Path to the trajectory file to open with ts.TorchSimTrajectory.
        property_name (str): Name of the property to retrieve. Must be one of the acceptable property names:
            - Properties recorded during simulation: "system_temperature", "system_potential", "slabwise_temperature"
            - Post-simulation processing: "v_hot_list", "v_cold_list", "v_hot_cumulative", "v_cold_cumulative",
            "dTdz_over_time", "dTdz_mean_z", "avg_dTdz", "final_thermal_conductivity"
            - Running statistics: "running_vhot_squared", "running_vcold_squared", "running_sum_vhot_squared",
            "running_sum_vcold_squared", "running_dTdz_over_time", "running_dTdz_mean_z",
            "running_avg_dTdz", "running_thermal_conductivity"

        Returns:
        numpy.ndarray: The array for the requested property as returned by traj.get_array(property_name).

        Logs the following error using Logger:
        KeyError: If property_name is not present in the acceptable property list (the function logs an error before raising).
        Raises:
        IOError: If the trajectory file cannot be opened or read.
        """
        # Check if the name exists in the acceptable_property_names
        acceptable_property_names = [
            # Properties Recorded During Simulation
            "system_temperature","system_potential" "slabwise_temperature"
            # Post-Simulation Processing
            'v_hot_list', 'v_cold_list', 'v_hot_cumulative', 'v_cold_cumulative','dTdz_over_time', 'dTdz_mean_z',
            'avg_dTdz', 'final_thermal_conductivity',
            # Running Statistics
            'running_vhot_squared', 'running_vcold_squared', 'running_sum_vhot_squared', 'running_sum_vcold_squared',
            'running_dTdz_over_time', 'running_dTdz_mean_z', 'running_avg_dTdz', 'running_thermal_conductivity']
        if str(property_name) not in acceptable_property_names:
            logger.error(f"Unable to find {property_name} property in acceptable_property_names")
        with ts.TorchSimTrajectory(filepath) as traj:
            system_simulation_prop = traj.get_array(property_name)
        return system_simulation_prop

    @staticmethod
    def simulation_file_store_property(
            filepath: PathLike,
            property_name,
            property_dataset,
            steps: list[int],
    ):
        """
        Write a single property to a Torch‑Sim trajectory file.

        The function accepts the property in many Python/NumPy/PyTorch forms,
        normalises it to a NumPy array with a leading frame axis that matches
        `len(steps)`, and stores it under `property_name` using
        `ts.TorchSimTrajectory` in append mode.

        Parameters
        ----------
        filepath (PathLike): Path to the ``.h5`` (or compatible) trajectory file.
        property_name (str): Key under which the property will be saved.
        property_dataset (list | tuple | np.ndarray | torch.Tensor | int | float):
            The data to store; can be a scalar, a 1‑D array, or a list of variable‑
            length tensors (cumulative data).
        steps list[int]: List of simulation step indices that the property corresponds to.
        """
        # ----------------------------------------------------------------------
        # Helper: pad a list of 1‑D tensors to a rectangular (N, L) tensor
        # ----------------------------------------------------------------------
        def pad_to_max(tensor_list, pad_value=0):
            """Return a (N, L) tensor where L = max length in the list."""
            max_len = max(t.numel() for t in tensor_list)          # longest length
            padded = torch.stack(
                [
                    torch.nn.functional.pad(
                        t, (0, max_len - t.numel()), value=pad_value
                    )
                    for t in tensor_list
                ]
            )
            return padded

        # ----------------------------------------------------------------------
        # Helper: convert any supported input into a mapping {name: ndarray}
        # ----------------------------------------------------------------------
        def make_single_mapping(
                prop_name: str,
                prop_dataset,
                steps: Union[int, list[int]],
        ):
            # Convert torch → NumPy; handle cumulative lists by padding first
            if isinstance(prop_dataset, torch.Tensor):
                arr = prop_dataset.detach().cpu().numpy()
            elif (
                    isinstance(prop_dataset, list)
                    and isinstance(prop_dataset[0], torch.Tensor)
                    and prop_dataset[0].shape != prop_dataset[-1].shape
            ):
                # Cumulative list → padded tensor → NumPy array
                prop_dataset = pad_to_max(prop_dataset)
                arr = prop_dataset.detach().cpu().numpy()
            else:
                arr = np.asarray(prop_dataset)

            # Expected number of frames
            n_frames = 1 if isinstance(steps, int) else len(steps)

            # Promote scalars to (1,)
            if arr.ndim == 0:
                arr = arr.reshape((1,))

            # Ensure a leading frame axis for single‑frame data
            if n_frames == 1 and arr.shape[0] != 1:
                arr = arr.reshape((1,) + arr.shape)

            # Replicate a single frame if multiple steps are requested
            if n_frames > 1:
                if arr.shape[0] == n_frames:
                    pass
                elif arr.shape[0] == 1:
                    arr = np.repeat(arr, n_frames, axis=0)
                else:
                    raise ValueError(
                        f"{prop_name}: first dim {arr.shape[0]} != n_frames {n_frames}"
                    )

            return {prop_name: arr}

        # ----------------------------------------------------------------------
        # Write the property to the trajectory file
        # ----------------------------------------------------------------------
        mapped_prop_dataset = make_single_mapping(property_name, property_dataset, steps)
        with ts.TorchSimTrajectory(filepath, mode="a") as traj:
            traj.write_arrays(mapped_prop_dataset, steps)


    # --------------------------------- End of Simulation File Functions ------------------------------------------ #


    # ------------------------------------------------------------------------------------------------------------- #
    #                       Functions to initialise, append and read data in intmd_file
    # ------------------------------------------------------------------------------------------------------------- #
    @staticmethod
    def intmd_file_read_all_data(filepath: PathLike="intmd_vexchange_data.h5", vhot: str= "vhot", vcold: str= "vcold"):
        """
        Read and return all data stored in the intermediate file i.e v_hot_list and v_cold_list
        (velocity exchange) information

        Parameters:
            filepath (PathLike): path to the HDF5 file.
            vhot (str): dataset name for hot values (default: 'vhot')
            vcold (str): dataset name for cold values (default: 'vcold')
        """
        with h5py.File(filepath, "r") as f:
            grp = f['data']
            vhot_values = grp[vhot][:]
            vcold_values = grp[vcold][:]

        return vhot_values, vcold_values # Both are (n_exchange_steps, n_exchange_per_step)

    @staticmethod
    def initialise_intmd_file(
            filepath: PathLike,
            n_store_calls: int,
            n_items_in_list: int,
            vhot_ds="vhot",
            vcold_ds="vcold",
            dtype=np.float32,
            mode: str = 'a',
            compression: Optional[str] = None) -> None:
        """
        Initialize datasets under the HDF5 group '/data' for preallocated, row-wise storage.
        Parameters:
            filepath (PathLike): path to the HDF5 file.
            n_store_calls (int): number of preallocated rows (first dimension).
            n_items_in_list (int): number of items per row (second dimension).
            vhot_ds (str): dataset name for hot values (default 'vhot').
            vcold_ds (str): dataset name for cold values (default 'vcold').
            dtype: NumPy dtype for storage (default np.float32).
            mode (str): file open mode passed to h5py.File (default 'a').
            compression (Optional[str]): compression filter name (e.g., 'gzip') or None.

        Behavior:
            Ensures the group '/data' exists in the HDF5 file at filepath.
            Ensures two datasets '/data/{vhot_ds}' and '/data/{vcold_ds}' exist with shape (n_store_calls, n_items_in_list) and dtype dtype.
            If an existing dataset has a different shape or dtype, it is deleted and recreated.
            Newly created datasets receive a persistent attribute 'write_idx' initialized to 0.
        """
        with h5py.File(filepath, mode) as f:
            # Creates a dataset group called 'data'
            grp = f.require_group('data')
            # Check if there's an already existing dataset with a different shape or dtype (to be deleted/overwritten)
            if vhot_ds in grp:
                old_dataset_vhot = grp[vhot_ds]
                if old_dataset_vhot.shape != (n_store_calls, n_items_in_list) or old_dataset_vhot.dtype != dtype:
                    logger.warning(f"Found existing dataset with different shape {old_dataset_vhot.shape} and/or dtype {old_dataset_vhot.dtype}.\nDeleting the existing dataset and creating a new one with shape {(n_store_calls, n_items_in_list)} and dtype {dtype}.")
                    # Due to the setup of both vcold and vhot data we know that vcold_ds is also not the same shape
                    del grp[vhot_ds], grp[vcold_ds]

            # Creates a dataset called 'vhot_ds'
            vhot_ds_obj = grp.require_dataset(vhot_ds,
                                              shape=(n_store_calls, n_items_in_list),
                                              dtype=dtype,
                                              chunks= (1, n_items_in_list),
                                              compression=compression,
                                              fillvalue=0)
            # Initialising 'write_idx' as a 'vhot_ds' dataset attribute
            vhot_ds_obj.attrs['write_idx'] = 0
            # Creates a dataset called 'vcold_ds'
            vcold_ds_obj = grp.require_dataset(vcold_ds,
                                               shape=(n_store_calls, n_items_in_list),
                                               dtype=dtype,
                                               chunks= (1, n_items_in_list),
                                               compression=compression,
                                               fillvalue=0)
            # Initialising 'write_idx' as a 'vcold_ds' dataset attribute
            vcold_ds_obj.attrs['write_idx'] = 0

    @staticmethod
    def append_to_intmd_file(
            vhot_vals: Sequence,
            vcold_vals: Sequence,
            filepath: PathLike,
            vhot_ds: str = "vhot",
            vcold_ds: str = "vcold",
            dtype=np.float32,
            mode: str = "a",
    ) -> Tuple[int, int]:
        """
        Append one row each to datasets '/data/{vhot_ds}' and '/data/{vcold_ds}' inside `filepath`.

        Parameters:
            vhot_vals, vcold_vals: 1-D sequences of length == n_items_in_list for their datasets.
            filepath: path to the HDF5 file.
            vhot_ds, vcold_ds: names of the datasets inside the '/data' group.
            dtype: numpy dtype to coerce input rows (default np.float32).
            mode: file open mode (default 'a').

        Behavior:
            Expects the datasets to already exist and be preallocated with shape
              (n_store_calls, n_items_in_list). The function reads the dataset shapes to
              validate input rows.
            Each dataset must contain an integer attribute 'write_idx' (defaults to 0
              if missing). The row is written at that index and 'write_idx' is incremented.
            Returns the new write indices (after the append) as (vhot_new_idx, vcold_new_idx).

        Logs the following error using Logger:
            KeyError: if a named dataset is missing under '/data'.
            ValueError: if input rows have incorrect shape.
            IndexError: if the target dataset is full.
        """
        vhot_vals_arr = np.asarray(vhot_vals, dtype=dtype)
        vcold_vals_arr = np.asarray(vcold_vals, dtype=dtype)

        with h5py.File(filepath, mode) as f:
            grp = f.require_group('data')  # ensures /data exists
            def get_data_and_write_idx(ds_name):
                dataset = grp.get(ds_name)
                if dataset is None:
                    logger.error(f"KeyError/Exception: Recieved an invalid dataset name, {ds_name}. Ensure you've initialized intmd_file appropriately.")
                n_store_calls, n_items_in_list = dataset.shape
                if vhot_vals_arr.shape != (n_items_in_list,) or vcold_vals_arr.shape != (n_items_in_list,):
                    logger.error(f"ValueError: rows must have shape ({n_items_in_list},). Recieved rows with dimensions: vhot {vhot_vals_arr.shape} | vcold {vcold_vals_arr.shape}.)")
                idx = int(dataset.attrs.get('write_idx', 0))
                if idx >= n_store_calls:
                    logger.error(f"IndexError: Store full for dataset '/data/{ds_name}' (idx {idx} >= {n_store_calls}).")
                return dataset, idx

            vhot_dataset, vhot_write_idx = get_data_and_write_idx(vhot_ds)
            vcold_dataset, vcold_write_idx = get_data_and_write_idx(vcold_ds)

            vhot_dataset[vhot_write_idx, :] = vhot_vals_arr
            vhot_dataset.attrs['write_idx'] = vhot_write_idx + 1

            vcold_dataset[vcold_write_idx, :] = vcold_vals_arr
            vcold_dataset.attrs['write_idx'] = vcold_write_idx + 1

        return vhot_write_idx + 1, vcold_write_idx + 1

    # ----------------------------------- End of intmd_file related Functions ------------------------------------- #

    # ------------------------------------------------------------------------------------------------------------- #
    #                                            Misc Functions:
    # ------------------------------------------------------------------------------------------------------------- #

    def _generate_system_slabs(self) -> None:
        """
        Create `nslabs` equally spaced slabs along the z‑axis of the simulation box.

        The method populates `self.lower` and `self.upper` with tensors of shape
        `(nslabs, 1)` containing the lower and upper z‑bounds of each slab, and also
        stores the slab thickness in `self.slab_length` (Å) and `self.slab_length_m` (metres).
        """
        # Build slab edges on the CPU, then move to the target device later
        slab_edges = np.linspace(0, self.z, self.nslabs + 1)   # (nslabs+1,)
        zlower_np = slab_edges[:-1]                                 # lower bounds
        zupper_np = slab_edges[1:]                                  # upper bounds
        slab_length = float((zupper_np - zlower_np).sum() / self.nslabs)

        # Convert NumPy arrays to column‑vector torch tensors
        lower = torch.from_numpy(zlower_np).unsqueeze(1)   # (nslabs, 1)
        upper = torch.from_numpy(zupper_np).unsqueeze(1)   # (nslabs, 1)

        # Store results on the instance
        self.slab_length = slab_length
        self.lower       = lower.to(self.simulation_device)
        self.upper       = upper.to(self.simulation_device)
        self.slab_length_m = slab_length * Angs_to_m


    def _formulate_cumulative_prop(self, final_property: torch.Tensor) -> list[torch.Tensor]:
        """
        Return a list of cumulative property tensors.

        Parameters:
            final_property (torch.Tensor): A 2‑D tensor. Values are assumed to be
              non‑negative; zeros are used as padding placeholders.

        Behavior:
            For each step i: The i‑th list element contains all values from 0 to i steps
              (trailing zeros are removed).
            The implementation uses a lower‑triangular mask and reshapes the result
              to avoid explicit Python loops.

        Returns:
            list[torch.Tensor]: A list where `result[i]` contains the cumulative values
              up to step i. Each tensor is a view of the original data, not a copy.
        """
        # Determine the dimensions of the property
        n_rows, n_columns = final_property.shape

        # Find the length of the largest possible array
        max_cumulative_len = n_rows * n_columns

        # Expand the final_property in order to pad it
        expanded_final_property = final_property.unsqueeze(0).expand(n_rows, n_rows, n_columns,)

        # Mask the values "progressively"
        idx_i = torch.arange(n_rows).unsqueeze(1)  # dim: (n_rows, 1)
        idx_j = torch.arange(n_rows).unsqueeze(0)  # dim: (1, n_columns)
        mask = (idx_j <= idx_i).unsqueeze(2)       # dim: (n_rows, n_columns, 1) bool

        # Use the make to not "count" preceding values:
        masked_final_property = expanded_final_property * mask.to(expanded_final_property.device)
        padded_cumulative_property = masked_final_property.reshape(n_rows, max_cumulative_len)
        cumulative_property = [prop[prop !=0.] for prop in padded_cumulative_property]
        return cumulative_property

    @staticmethod
    def _configure_logger(level=logging.INFO, logfile=None) -> None:
        """
         Set up the root logger with a console handler and, optionally, a rotating file handler.

         Parameters
            level : int, optional
                Logging level (default ``logging.INFO``).
            logfile : str or None, optional
                Path to a log file; if provided a ``RotatingFileHandler`` (10 KB, 5 backups) is added.
        """
        root = logging.getLogger()
        root.setLevel(level)
        # remove default handlers if reconfiguring
        for h in list(root.handlers):
            root.removeHandler(h)

        fmt = "(%(asctime)s) %(name)s: %(message)s"
        date_fmt = "%m-%d %H:%M"
        formatter = logging.Formatter(fmt, date_fmt)

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        root.addHandler(console)

        if logfile is not None:
            fh = RotatingFileHandler(logfile, maxBytes=10000, backupCount=5)
            fh.setFormatter(formatter)
            root.addHandler(fh)
        return None
    # --------------------------------- End of Misc. File Functions ------------------------------------------ #
