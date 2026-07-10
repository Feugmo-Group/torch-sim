import os
from os import PathLike
import torch
import numpy as np
import torch_sim as ts
from typing import Sequence, Optional, Tuple, Union, Mapping, Callable, Literal, get_args
import logging
from logging.handlers import RotatingFileHandler
import sys
from torch_sim import fire_init, fire_step, SimState, FireState, OptimState
from torch_sim.integrators import MDState
from torch_sim.models.interface import ModelInterface
from torch_sim.autobatching import calculate_memory_scalers, estimate_max_memory_scaler


class ImplementationBase():
    def __init__(
            self,
            *,
            method: Literal["EMD", "GMKA", "HNEMD", "HNEMDEC", "RNEMD"],
            logger: logging.Logger,
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
            display_pbar=False,
            log_to_file: bool=True,
    ):
        self.method = method
        self.logger = logger
        # Initializing Simulation Parameters:
        self.system_state, self.device, self.dtype = system_state, system_state.device, system_state.dtype
        self.temp_K = temp_K
        self.simulation_timestep = timestep_ps
        self.nsteps_total = nsteps_total
        self.model = model
        self.n_simulations = n_simulations
        self.model_memory_scaling = model_memory_scaling

        # Initializing Equilibration Parameters:
        self.perform_equilibration = perform_equilibration
        self.n_equilibration_steps = n_equilibration_steps
        self.store_equilibration_data = store_equilibration_data
        self.equilibration_quantities_functions = equilibration_quantities_functions
        self.equilibration_data_filename = equilibration_data_filename

        # Initializing Geometry Optimization Parameters:
        self.perform_geometry_optimization = perform_geometry_optimization
        self.geomtery_optimizer = geometry_optimizer
        self.optimizer_convergence_fn = optimizer_convergence_fn

        # Initializing Data and log files parameters:
        self.display_pbar = display_pbar
        if data_folder_path_abs is None:
            # Get the current working directory (from where EMD is being called)
            directory = os.getcwd()
            data_folder_path_abs = os.path.join(directory, f'{method}_simulation_data')
        os.makedirs(data_folder_path_abs, exist_ok=True)
        self.data_folder_path_abs = data_folder_path_abs

        if log_to_file:
            self._configure_logger(logfile=os.path.join(self.data_folder_path_abs, 'simulation_logs.log'))
            logger.info(f"Logging to {os.path.join(self.data_folder_path_abs, 'simulation_logs.log')}")
        else:
            logger.info("Logger initialised to not log to a file")

        self.simulation_data_filename = simulation_data_filename
        self.equilibration_filepath = [os.path.join(self.data_folder_path_abs, f"{equilibration_data_filename}-system-{i}.h5") for i in range(1,n_simulations+1)]
        self.simulation_filepath= [os.path.join(self.data_folder_path_abs, f"{simulation_data_filename}-system-{i}.h5") for i in range(1, n_simulations+1)]

        # Initializing autobatcher(s) for batch simulations:
        if self.system_state.n_systems != self.n_simulations:
            self.system_state = ts.initialize_state([self.system_state] * self.n_simulations, device=self.device, dtype=self.dtype)

        # ---
        # Prints/Outputs Model Memory Estimation logs
        memory_metric_values = calculate_memory_scalers(self.system_state, self.model_memory_scaling)
        max_memory_scalar = estimate_max_memory_scaler(self.system_state, self.model, metric_values=memory_metric_values)
        # ---
        self.inflight_batcher = ts.InFlightAutoBatcher(
            model=self.model, memory_scales_with=self.model_memory_scaling ,
            max_memory_scaler = max_memory_scalar,
        )
        self.binning_batcher = ts.BinningAutoBatcher(
            model=self.model, memory_scales_with=self.model_memory_scaling ,
            max_memory_scaler= max_memory_scalar,
        )


    # @staticmethod
    def _configure_logger(self, level=logging.INFO, logfile=None) -> None:
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

        fmt = "(%(asctime)s) %(name)s: %(message)s" # f"(%(asctime)s) {self.method} (%(name)s): %(message)s"
        date_fmt = "%m-%d %H:%M"
        formatter = logging.Formatter(fmt, date_fmt)

        # Handler for stdout (to handle INFO and lower levels)
        stdout_console = logging.StreamHandler(sys.stdout)
        stdout_console.setFormatter(formatter)
        stdout_console.setLevel(level)

        # Handler for stderr (to handle WARING and higher levels)
        stderr_console = logging.StreamHandler(sys.stderr)
        stderr_console.setFormatter(formatter)
        stderr_console.setLevel(logging.WARNING)

        root.addHandler(stdout_console)
        root.addHandler(stderr_console)
        if logfile is not None:
            fh = RotatingFileHandler(logfile, maxBytes=10000, backupCount=5)
            fh.setFormatter(formatter)
            root.addHandler(fh)
        return None

    # ------------------------------------------------------------------------------------------------------------- #
    #                                      Pre-Simulation Functions:
    # ------------------------------------------------------------------------------------------------------------- #
    @staticmethod
    def _perform_geometry_optimization(
            method_instance,
            logger: logging.Logger=None,
            ) -> OptimState:
        logger.info("Performing Geometry Optimization...")
        # Optimize/Relax the geometry of the SimulationBox:
        # i.e. minimize the potential energy and the forces in the system
        def custom_convergence_fn(state, last_energy):
            energy_convergence_fn = ts.runners.generate_energy_convergence_fn(energy_tol=1e-6)
            force_convergence_fn = ts.runners.generate_force_convergence_fn(force_tol=1e-6, include_cell_forces=False)
            energy_output = energy_convergence_fn(state, last_energy)
            force_output = force_convergence_fn(state, last_energy)
            return torch.logical_and(energy_output, force_output)

        relaxed_state = ts.runners.optimize(
            system=method_instance.system_state,
            model=method_instance.model,
            optimizer= method_instance.geomtery_optimizer,
            convergence_fn= method_instance.optimizer_convergence_fn if method_instance.optimizer_convergence_fn else custom_convergence_fn,
            autobatcher=method_instance.inflight_batcher,
            pbar={'ascii': True, "desc": 'Optimization/Relaxation: ', 'colour': 'magenta'} if method_instance.display_pbar else None,
        )
        logger.info(f"[Geometry optimization finished] Post-Relaxation Energy is {method_instance.model(relaxed_state)['energy'].tolist()} eV")
        return relaxed_state

    @staticmethod
    def _perform_equilibration(
            method_instance,
            logger: logging.Logger=None,
    ) -> MDState:
        """
        Run equilibration integration on the current system_state using the stored simulation parameters.

        Behavior:
            Reads equilibration-related parameters from self.simulation_parameters.
            Registers property calculators with a TrajectoryReporter if store_equil_data is enabled.
            Updates self.system_state with final MDState returned by the integrator.
        """
        logger.info("Performing Equilibration...")
        basic_props = {1: {"system_temperature": ImplementationBase._calculate_system_temperature,
                           "system_potential": ImplementationBase._calculate_system_potential}
                       }
        n_equil_step = method_instance.n_equilibration_steps
        store_equil_data = method_instance.store_equilibration_data
        equi_prop_to_record = basic_props | ({} if not isinstance(method_instance.equilibration_quantities_functions, dict) else method_instance.equilibration_quantities_functions)

        if store_equil_data:
            equilibrium_file_reporter = ts.TrajectoryReporter(
                filenames=method_instance.equilibration_filepath,
                prop_calculators=equi_prop_to_record,
            )
            logger.info(f"Equilibration Data filepath set to {method_instance.equilibration_filepath}")
        else:
            equilibrium_file_reporter = None

        system_state= ts.integrate(
            system = method_instance.system_state,
            model  = method_instance.model,
            integrator = ts.Integrator.nvt_nose_hoover,
            n_steps = n_equil_step,
            timestep = method_instance.simulation_timestep,
            temperature = method_instance.temp_K,
            trajectory_reporter = equilibrium_file_reporter,
            autobatcher= method_instance.binning_batcher,
            pbar={'ascii': True, "desc": 'Equilibration: ', 'colour': 'magenta'} if method_instance.display_pbar else None,
        )
        sys_temp_k = ts.quantities.calc_temperature(masses=system_state.masses,
                                                    momenta=system_state.momenta,
                                                    )
        logger.info(f"[Equilibration Completed] Post-Equilibration System Temperature: {sys_temp_k:.2f} K")
        if torch.isnan(system_state.positions).any():
            logger.error(f"NaN values in system_MDState.positions after equilibration")
        return system_state

    @staticmethod
    def _calculate_system_temperature(state: MDState) -> torch.Tensor:
        return ts.quantities.calc_temperature(masses=state.masses, momenta=state.momenta)

    @staticmethod
    def _calculate_system_potential(state: MDState, model: ModelInterface) -> torch.Tensor:
        return model(state)['energy']

implemented_methods = Literal["RNEMD", "EMD", "GMKA", "HNEMD", "HNEMDEC"]
acceptable_property_names = {
    "RNEMD": [],
    "EMD": [],
    "GMKA": [],
    "HNEMD": [],
    "HNEMDEC": [],
}
class DataSetIO:
    @staticmethod
    def store_property(
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

    def load_property(self, filepath: PathLike, property_name:str):
        """
        Reads the simulation file and returns the property dataset
        Retrieve a named property array from a simulation trajectory file.
        Parameters:
        filepath (PathLike):  Path to the trajectory file to open with ts.TorchSimTrajectory.
        property_name (str): Name of the property to retrieve. Must be one of the acceptable property names:

        Returns:
        numpy.ndarray: The array for the requested property as returned by traj.get_array(property_name).

        Note: The function doesn't check whether the property_name is valid.
        """
        with ts.TorchSimTrajectory(filepath) as traj:
            system_simulation_prop = traj.get_array(property_name)
        return system_simulation_prop


class WIP():
    def __init__(self):
        print("WIP")