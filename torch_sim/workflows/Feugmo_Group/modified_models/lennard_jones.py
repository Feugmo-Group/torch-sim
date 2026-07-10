"""Classical pairwise interatomic potential model.

This module implements the Lennard-Jones potential for molecular dynamics simulations.
It provides efficient calculation of energies, forces, and stresses based on the
classic 12-6 potential function. The implementation supports both full pairwise
calculations and neighbor list-based optimizations.

Example::

    # Create a Lennard-Jones model with default parameters
    model = LennardJonesModel(device=torch.device("cuda"))

    # Create a model with custom parameters
    model = LennardJonesModel(
        sigma=3.405,  # Angstroms
        epsilon=0.01032,  # eV
        cutoff=10.0,  # Angstroms
        compute_stress=True,
    )

    # Calculate properties for a simulation state
    output = model(sim_state)
    energy = output["energy"]
    forces = output["forces"]
"""
import warnings
import torch
from typing import Callable
import torch_sim as ts
from torch_sim import transforms
from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import torchsim_nl
from torch_sim.typing import StateLike
from Feugmo_Group.DFTD3_corrections import DFTD3Corrections

DEFAULT_SIGMA = 1.0
DEFAULT_EPSILON = 1.0

def calculate_external_force(energies, momenta, masses, virials, Fe_vec):
    "F_ext or driving force for HNEMD"
    momenta_norm = torch.norm(momenta, dim=-1)
    momenta_squared = momenta_norm * momenta_norm
    term_1 = torch.einsum("b, n ->  bn", (0.5 * momenta_squared/masses + energies), Fe_vec)
    term_2 = torch.mean(term_1, dim=0)
    term_3 = torch.einsum("bmn,n->bm", virials, Fe_vec)
    term_4 = torch.mean(term_3, dim=0)
    return term_1 + term_3 - (term_2 + term_4)

def per_atom_virial(dr_vec , sigma, epsilon, cutoff, state, mapping, device, dtype):
    dr_vec.requires_grad_(True)
    dr = torch.linalg.norm(dr_vec, ord=2, dim=-1)
    idr6 = torch.pow(sigma/dr, 6)
    idr12 = torch.pow(sigma/dr, 12)

    # Calculate potential energy
    energy = 4.0 * epsilon * (idr12 - idr6)
    energy = torch.where(dr > 0, energy, torch.zeros_like(energy))
    # print("This is energy:", energy, energy.shape)

    # Combine masks and use in-place multiplication
    mask = (dr > 0) & (dr < cutoff) # handling instablities and zeroing out energies beyond cutoff
    pair_energies = (energy * mask.float())  # Faster than torch.where
    # deleting non-required variable:
    del idr12, idr6, energy , dr
    # print("This is the shape of pair energies: ", pair_energies.shape)
    positions = state.positions
    atom_energies = torch.zeros(
        positions.shape[0], dtype=dtype, device=device,
    )
    # Each atom gets half of the pair energy
    atom_energies.index_add_(0, mapping[0], 0.5 * pair_energies.float())
    atom_energies.index_add_(0, mapping[1], 0.5 * pair_energies.float())
    # print("This is atom_energies: ", atom_energies, atom_energies.shape)

    with torch.no_grad():
        dUdr_like = torch.ones_like(atom_energies)
        dUdr_term = torch.autograd.grad(
            outputs=atom_energies,
            inputs=dr_vec,
            grad_outputs=dUdr_like,
            retain_graph=False,
            create_graph=False,
        )[0]
        del dUdr_like

        # Outer product using einsum
        term_tensor = torch.einsum('bi,bj->bij', dUdr_term, -1*dr_vec)

        # Create output tensor
        atom_virial = torch.zeros((state.positions.shape[0], 3, 3), dtype=term_tensor.dtype, device=device)


        # Expand mapping to match term_tensor shape
        # mapping tells us which "bin" each term_tensor entry goes into
        mapping_expanded = mapping[0].view(-1, 1, 1).expand_as(term_tensor)
        # Use scatter_add_ to sum term_tensor entries by mapping indices
        atom_virial.scatter_add_(0, mapping_expanded, term_tensor)

        total_virial = torch.sum(atom_virial, dim=0)
        del dr_vec, dUdr_term, term_tensor
        if device == torch.device('cuda'):
            torch.cuda.empty_cache()
        pair_energies = pair_energies.detach()
        atom_energies = atom_energies.detach()
        total_virial = total_virial.detach()
        atom_virial = atom_virial.detach()
        # dr_vec.detach()
        return pair_energies , atom_energies, total_virial, atom_virial

def lennard_jones_pair(
        dr: torch.Tensor,
        sigma: float | torch.Tensor = DEFAULT_SIGMA,
        epsilon: float | torch.Tensor = DEFAULT_EPSILON,
) -> torch.Tensor:
    """Calculate pairwise Lennard-Jones interaction energies between particles.

    Implements the standard 12-6 Lennard-Jones potential that combines short-range
    repulsion with longer-range attraction. The potential has a minimum at r=sigma.

    The functional form is:
    V(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]

    Args:
        dr: Pairwise distances between particles. Shape: [n, m].
        sigma: Distance at which potential reaches its minimum. Either a scalar float
            or tensor of shape [n, m] for particle-specific interaction distances.
        epsilon: Depth of the potential well (energy scale). Either a scalar float
            or tensor of shape [n, m] for pair-specific interaction strengths.

    Returns:
        torch.Tensor: Pairwise Lennard-Jones interaction energies between particles.
            Shape: [n, m]. Each element [i,j] represents the interaction energy between
            particles i and j.
    """
    # Calculate inverse dr and its powers
    idr = sigma / dr
    idr2 = idr * idr
    idr6 = idr2 * idr2 * idr2
    idr12 = idr6 * idr6

    # Calculate potential energy
    energy = 4.0 * epsilon * (idr12 - idr6)

    # Handle potential numerical instabilities and infinities
    return torch.where(dr > 0, energy, torch.zeros_like(energy))
    # return torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)


def lennard_jones_pair_force(
        dr: torch.Tensor,
        sigma: float | torch.Tensor = DEFAULT_SIGMA,
        epsilon: float | torch.Tensor = DEFAULT_EPSILON,
) -> torch.Tensor:
    """Calculate pairwise Lennard-Jones forces between particles.

    Implements the force derived from the 12-6 Lennard-Jones potential. The force
    is repulsive at short range and attractive at long range, with a zero-crossing
    at r=sigma.

    The functional form is:
    F(r) = 24*epsilon/r * [(2*sigma^12/r^12) - (sigma^6/r^6)]

    This is the negative gradient of the Lennard-Jones potential energy.

    Args:
        dr: Pairwise distances between particles. Shape: [n, m].
        sigma: Distance at which force changes from repulsive to attractive.
            Either a scalar float or tensor of shape [n, m] for particle-specific
            interaction distances.
        epsilon: Energy scale of the interaction. Either a scalar float or tensor
            of shape [n, m] for pair-specific interaction strengths.

    Returns:
        torch.Tensor: Pairwise Lennard-Jones forces between particles. Shape: [n, m].
            Each element [i,j] represents the force magnitude between particles i and j.
            Positive values indicate repulsion, negative values indicate attraction.
    """
    # Calculate inverse dr and its powers
    idr = sigma / dr
    idr2 = idr * idr
    idr6 = idr2 * idr2 * idr2
    idr12 = idr6 * idr6

    # Calculate force (negative gradient of potential)
    # F = -24*epsilon/r * ((sigma/r)^6 - 2*(sigma/r)^12)
    force = 24.0 * epsilon / dr * (2.0 * idr12 - idr6)

    # Handle potential numerical instabilities and infinities
    return torch.where(dr > 0, force, torch.zeros_like(force))


class LennardJonesModel(ModelInterface):
    """Lennard-Jones potential energy and force calculator.

    Implements the Lennard-Jones 12-6 potential for molecular dynamics simulations.
    This model calculates pairwise interactions between atoms and supports either
    full pairwise calculation or neighbor list-based optimization for efficiency.

    Attributes:
        sigma (torch.Tensor): Length parameter controlling particle size/repulsion
            distance.
        epsilon (torch.Tensor): Energy parameter controlling interaction strength.
        cutoff (torch.Tensor): Distance cutoff for truncating potential calculation.
        device (torch.device): Device where calculations are performed.
        dtype (torch.dtype): Data type used for calculations.
        compute_forces (bool): Whether to compute atomic forces.
        compute_stress (bool): Whether to compute stress tensor.
        per_atom_energies (bool): Whether to compute per-atom energy decomposition.
        per_atom_stresses (bool): Whether to compute per-atom stress decomposition.
        use_neighbor_list (bool): Whether to use neighbor list optimization.

    Example::

        # Basic usage with default parameters
        lj_model = LennardJonesModel(device=torch.device("cuda"))
        results = lj_model(sim_state)

        # Custom parameterization for Argon
        ar_model = LennardJonesModel(
            sigma=3.405,  # Å
            epsilon=0.0104,  # eV
            cutoff=8.5,  # Å
            compute_stress=True,
        )
    """

    def __init__(
            self,
            sigma: float = 1.0,
            epsilon: float = 1.0,
            device: torch.device | None = None,
            dtype: torch.dtype = torch.float32,
            *,  # Force keyword-only arguments
            compute_forces: bool = True,
            compute_stress: bool = False,
            compute_virial: bool = False,
            per_atom_energies: bool = False,
            per_atom_stresses: bool = False,
            use_neighbor_list: bool = True,
            cutoff: float | None = None,
            compute_external_force: bool = False,
            external_force_fn: Callable=None,
            Fe_vec: torch.Tensor= None,
            dftd3_corrections: DFTD3Corrections = None,
    ) -> None:
        """Initialize the Lennard-Jones potential calculator.

        Creates a model with specified interaction parameters and computational flags.
        The model can be configured to compute different properties (forces, stresses)
        and use different optimization strategies.

        Args:
            sigma (float): Length parameter of the Lennard-Jones potential in distance
                units. Controls the size of particles. Defaults to 1.0.
            epsilon (float): Energy parameter of the Lennard-Jones potential in energy
                units. Controls the strength of the interaction. Defaults to 1.0.
            device (torch.device | None): Device to run computations on. If None, uses
                CPU. Defaults to None.
            dtype (torch.dtype): Data type for calculations. Defaults to torch.float32.
            compute_forces (bool): Whether to compute forces. Defaults to True.
            compute_stress (bool): Whether to compute stress tensor. Defaults to False.
            compute_virial (bool): Whether to compute per-atom and system virial tensor. Defaults to False.
            compute_external_force (bool): Whether to compute driving force for HNEMD. Defaulta to False.
            Fe_vec (torch.Tensor): Fe vector (size=(3,)) for HNEMD simulations
            per_atom_energies (bool): Whether to compute per-atom energy decomposition.
                Defaults to False.
            per_atom_stresses (bool): Whether to compute per-atom stress decomposition.
                Defaults to False.
            use_neighbor_list (bool): Whether to use a neighbor list for optimization.
                Significantly faster for large systems. Defaults to True.
            cutoff (float | None): Cutoff distance for interactions in distance units.
                If None, uses 2.5*sigma. Defaults to None.

        Example::

            # Model with custom parameters
            model = LennardJonesModel(
                sigma=3.405,
                epsilon=0.01032,
                device=torch.device("cuda"),
                dtype=torch.float64,
                compute_stress=True,
                per_atom_energies=True,
                cutoff=10.0,
            )
        """
        super().__init__()
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_external_force = compute_external_force
        self._compute_stress = compute_stress
        self._compute_virial = compute_virial
        self.per_atom_energies = per_atom_energies
        self.per_atom_stresses = per_atom_stresses
        self.use_neighbor_list = use_neighbor_list
        if compute_external_force:
            if not isinstance(external_force_fn, Callable):
                raise TypeError("external_force_fn should be a Callable")
            else:
                self.external_force_fn = external_force_fn
            if not per_atom_energies or not compute_forces:
                # Need them for external forces calculation
                UserWarning("Compute external force was set to True but either per_atom_energies or compute_forces wasn't set to True. Overriding them to be True")
                self._compute_forces = True
                self.per_atom_energies = True
            if Fe_vec is not None and isinstance(Fe_vec, torch.Tensor) and Fe_vec.shape == (3,):
                self.Fe_vec = Fe_vec
            else:
                raise ValueError("Compute external force was set to True but Fe_vec wasn't provided or Fe_vec wasn't a torch.Tensor of shape (3,)")



        # Convert parameters to tensors
        self.sigma = torch.tensor(sigma, dtype=dtype, device=self.device)
        self.cutoff = torch.tensor(cutoff or 2.5 * sigma, dtype=dtype, device=self.device)
        self.epsilon = torch.tensor(epsilon, dtype=dtype, device=self.device)

        if isinstance(dftd3_corrections, DFTD3Corrections):
            self.dftd3_corrections = dftd3_corrections
            self.compute_correction = True
        else:
            if dftd3_corrections is None:
                self.compute_correction = False
            else:
                raise ValueError(f"'dftd3_corrections': Expected DFTD3Corrections class got {type(dftd3_corrections)}")


    def unbatched_forward(
            self,
            state: ts.SimState,
    ) -> dict[str, torch.Tensor]:
        """Compute Lennard-Jones properties for a single unbatched system.

        Internal implementation that processes a single, non-batched simulation state.
        This method handles the core computations of pair interactions, neighbor lists,
        and property calculations.

        Args:
            state (SimState): Single, non-batched simulation state containing atomic
                positions, cell vectors, and other system information.

        Returns:
            dict[str, torch.Tensor]: Computed properties:
                - "energy": Total potential energy (scalar)
                - "forces": Atomic forces with shape [n_atoms, 3] (if
                    compute_forces=True)
                - "stress": Stress tensor with shape [3, 3] (if compute_stress=True)
                - "energies": Per-atom energies with shape [n_atoms] (if
                    per_atom_energies=True)
                - "stresses": Per-atom stresses with shape [n_atoms, 3, 3] (if
                    per_atom_stresses=True)

        Notes:
            This method handles two different approaches:
            1. Neighbor list approach: Efficient for larger systems
            2. Full pairwise calculation: Better for small systems

            The implementation applies cutoff distance to both approaches for consistency.
        """
        if not isinstance(state, ts.SimState):
            state = ts.SimState(**state)

        positions = state.positions
        cell = state.row_vector_cell
        cell = cell.squeeze()
        pbc = state.pbc

        # Ensure system_idx exists (create if None for single system)
        system_idx = (
            state.system_idx
            if state.system_idx is not None
            else torch.zeros(positions.shape[0], dtype=torch.long, device=self.device)
        )

        # Wrap positions into the unit cell
        wrapped_positions = (
            ts.transforms.pbc_wrap_batched(positions, state.cell, system_idx, pbc)
            if pbc.any()
            else positions
        )

        if self.use_neighbor_list:
            mapping, _, shifts_idx = torchsim_nl(
                positions=wrapped_positions,
                cell=cell,
                pbc=pbc,
                cutoff=self.cutoff,
                system_idx=system_idx,
            )
            # Pass shifts_idx directly - get_pair_displacements will convert them
            dr_vec, distances = transforms.get_pair_displacements(
                positions=wrapped_positions,
                cell=cell,
                pbc=pbc,
                pairs=(mapping[0], mapping[1]),
                shifts=shifts_idx,
            )
        else:
            # Get all pairwise displacements
            dr_vec, distances = transforms.get_pair_displacements(
                positions=wrapped_positions, cell=cell, pbc=pbc
            )
            # Mask out self-interactions
            mask = torch.eye(
                wrapped_positions.shape[0], dtype=torch.bool, device=self.device
            )
            distances = distances.masked_fill(mask, float("inf"))
            # Apply cutoff
            mask = distances < self.cutoff
            # Get valid pairs - match neighbor list convention for pair order
            i, j = torch.where(mask)
            mapping = torch.stack([j, i])
            # Get valid displacements and distances
            dr_vec = dr_vec[mask]
            distances = distances[mask]

        # Mask for radial cutoff
        mask = distances < self.cutoff

        # Calculate per-atom virial
        if self._compute_virial:
            pair_energies, atom_energies, total_virial, atom_virial = per_atom_virial(dr_vec, self.sigma, self.epsilon, self.cutoff, state, mapping, self.device, self.dtype)
            results = {"energy": 0.5 * pair_energies.sum(),
                       'virial': total_virial, 'virials': atom_virial}
            if self.per_atom_energies:
                results['energies'] = atom_energies

            dr_vec, distances = dr_vec.detach(), distances.detach()
        else:
            # Calculate pair energies and apply cutoff
            pair_energies = lennard_jones_pair(
                distances, sigma=self.sigma, epsilon=self.epsilon
            )
            # Zero out energies beyond cutoff
            pair_energies = torch.where(mask, pair_energies, torch.zeros_like(pair_energies))

            # Initialize results with total energy (sum/2 to avoid double counting)
            results = {"energy": 0.5 * pair_energies.sum()}

            if self.per_atom_energies:
                atom_energies = torch.zeros(
                    positions.shape[0], dtype=self.dtype, device=self.device
                )
                # Each atom gets half of the pair energy
                atom_energies.index_add_(0, mapping[0], 0.5 * pair_energies.float())
                atom_energies.index_add_(0, mapping[1], 0.5 * pair_energies.float())
                results["energies"] = atom_energies

        if self.compute_forces or self.compute_stress:
            # Calculate forces and apply cutoff
            pair_forces = lennard_jones_pair_force(
                distances, sigma=self.sigma, epsilon=self.epsilon
            )
            pair_forces = torch.where(mask, pair_forces, torch.zeros_like(pair_forces))

            # Project forces along displacement vectors
            force_vectors = (pair_forces / distances)[:, None] * dr_vec

            if self.compute_forces:
                # Initialize forces tensor
                forces = torch.zeros_like(positions)
                # Add force contributions (f_ij on i, -f_ij on j)
                forces.index_add_(0, mapping[0], -force_vectors)
                forces.index_add_(0, mapping[1], force_vectors)
                results["forces"] = forces

            if self.compute_stress and cell is not None:
                # Compute stress tensor
                stress_per_pair = torch.einsum("...i,...j->...ij", dr_vec, force_vectors)
                volume = torch.abs(torch.linalg.det(cell))

                results["stress"] = -stress_per_pair.sum(dim=0) / volume

                if self.per_atom_stresses:
                    atom_stresses = torch.zeros(
                        (state.positions.shape[0], 3, 3),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    atom_stresses.index_add_(0, mapping[0], -0.5 * stress_per_pair.float())
                    atom_stresses.index_add_(0, mapping[1], -0.5 * stress_per_pair.float())
                    results["stresses"] = atom_stresses / volume

        if self.compute_corrections:
            energy_correction, forces_correction, virial_correction = self.dftd3_corrections.compute_corrections(state_positions=wrapped_positions, state_cell=state.cell,
                                                                                                                 system_idx=state.system_idx, unit_shifts=shifts_idx,
                                                                                                                 state_atomic_numbers=state.atomic_numbers)
            results['energy'] = results['energy'] + energy_correction
            results['forces'] = results['forces'] + forces_correction
            results['virials'] = results['virials'] + virial_correction
        if self._compute_external_force:
            if isinstance(state, ts.integrators.MDState): # only compute when doing Molecular Dynamics
                external_force = self.external_force_fn(atom_energies, state.momenta, state.masses, results['virials'], self.Fe_vec)
                results['forces'] = (forces + external_force) - torch.mean(external_force, dim=0)

        return results

    def forward(self, state: ts.SimState | StateLike) -> dict[str, torch.Tensor]:
        """Compute Lennard-Jones energies, forces, and stresses for a system.

        Main entry point for Lennard-Jones calculations that handles batched states by
        dispatching each system to the unbatched implementation and combining results.

        Args:
            state (SimState | StateDict): Input state containing atomic positions,
                cell vectors, and other system information. Can be a SimState object
                or a dictionary with the same keys.

        Returns:
            dict[str, torch.Tensor]: Computed properties:
                - "energy": Potential energy with shape [n_systems]
                - "forces": Atomic forces with shape [n_atoms, 3] (if
                    compute_forces=True)
                - "stress": Stress tensor with shape [n_systems, 3, 3] (if
                    compute_stress=True)
                - "energies": Per-atom energies with shape [n_atoms] (if
                    per_atom_energies=True)
                - "stresses": Per-atom stresses with shape [n_atoms, 3, 3] (if
                    per_atom_stresses=True)

        Raises:
            ValueError: If system cannot be inferred for multi-cell systems.

        Example::

            # Compute properties for a simulation state
            model = LennardJonesModel(compute_stress=True)
            results = model(sim_state)

            energy = results["energy"]  # Shape: [n_systems]
            forces = results["forces"]  # Shape: [n_atoms, 3]
            stress = results["stress"]  # Shape: [n_systems, 3, 3]
            energies = results["energies"]  # Shape: [n_atoms]
            stresses = results["stresses"]  # Shape: [n_atoms, 3, 3]
        """
        sim_state = (
            state
            if isinstance(state, ts.SimState)
            else ts.SimState(**state, masses=torch.ones_like(state["positions"]))
        )

        if sim_state.system_idx is None and sim_state.cell.shape[0] > 1:
            raise ValueError("System can only be inferred for batch size 1.")

        outputs = [
            self.unbatched_forward(sim_state[idx]) for idx in range(sim_state.n_systems)
        ]
        properties = outputs[0]

        # we always return tensors
        # per atom properties are returned as (atoms, ...) tensors
        # global properties are returned as shape (..., n) tensors
        results: dict[str, torch.Tensor] = {}
        for key in ("stress", "energy", 'virial'):
            if key in properties:
                results[key] = torch.stack([out[key] for out in outputs])
        for key in ("forces", "energies", "stresses", 'virials'):
            if key in properties:
                results[key] = torch.cat([out[key] for out in outputs], dim=0)

        return results
