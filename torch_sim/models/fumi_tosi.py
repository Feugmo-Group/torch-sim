""" Born-Mayer-Huggins / Fumi-Tosi potential Model.

This module implements the Fumi-Tosi potential for molecular dynamics simulations.
It provides efficient calculation of energy, forces, and stresses based on the
Fumi-Tosi potential function. The implementation supports both full pairwise
calculations and neighbor list-based optimizations.

Example::
    a = torch.tensor([0.421999990940094, 0.29010000824928284, 0.1581999957561493])
    c = torch.tensor([0.045570001006126404, 1.2484999895095825, 69.29000091552734])
    d = torch.tensor([0.018727000802755356, 1.4982000589370728, 139.2100067138672])
    model = FumiTosiModel(
        atomic_number_zi=3,                                # Atomic number of Li
        ionic_charge_i= 1,                                 # ionic charge of Li
        ionic_charge_j= -1,                                # ionic charge of Cl
        sigma_ij= torch.tensor([1.632, 2.401, 3.170]),     # sigma_ij (in Å) for Li-Li, Li-Cl & Cl-Cl pairs
        a=a,                                               # Aij (in eV)
        b=torch.tensor([2.9200]),                          # B (in Å^(-1))
        c=c,                                               # Cij (in eV * Å^(6))
        d=d,                                               # Dij (in eV * Å^8)
        rc=7,                                              # radial cutoff (in Å)
        device=torch.device('cuda'),
        dtype=torch.float32,                               # defaults to torch.float32
        compute_forces=True,                               # True by default
        compute_stress=True,
        per_atom_energies=True,                            # True by default
        per_atom_stresses=True,
        use_neighbor_list=True,                            # True by default
    )

    # Calculate properties for a simulation state
    output = model(sim_state)
    energy = output["energy"]
    forces = output["forces"]
"""

import torch
import torch_sim as ts
from torch_sim import transforms
from torch_sim.state import SimState
from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import torchsim_nl
from torch_sim.typing import StateDict

e = 1.602_176_634e-19  # elementary charge (SI units)
DEFAULT_Rc = 7 # cutoff radius

def tomi_fusi_pair(
        dr: torch.Tensor,
        pair_mask: torch.Tensor,
        ionic_charge_i: int | torch.Tensor,
        ionic_charge_j: int | torch.Tensor,
        sigma_ij: float | torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
) -> torch.Tensor:
    """Calculate pairwise Fumi-Tosi potential (interaction) energies between particles

    Implements the Fumi-Tosi potential for molecular dynamics simulations.

    The functional form is:
    U(r) = k * (e**2 * zi*zj)/r + Aij * exp(B *(sigma_ij -r)) - Cij/r^6 - Dij/r^8    # B = 1/rho

    Args:
        dr: Pairwise distances between particles. shape: [n,m]
        pair_mask: Unordered pair index mapping for two species
            0: (zi, zi); 1: (zi, zj) or (zj, zi); 2: (zj, zj) pairs
        ionic_charge_i: Charge number of chemical species i e.g +1, -2
        ionic_charge_j: Charge number of chemical species j e.g -2, +1
        sigma_ij: Born-Mayer-Huggins repulsive parameter (Interaction-dependent length parameter)
        a: Aij parameter (Born-Mayer-Huggins-Tosi-Fumi repulsive parameter)
        b: B parameter (1/rho) (Born-Mayer-Huggings-Tosi-Fumi repulsive parameter)
        c: Cij parameter (Van der Waals parameter)
        d: Dij parameter (Van der Waals parameter)

    Returns:
          torch.Tensor: Pairwise Fumi-Tosi potential energies between particles.
              Shape: [n,m]. Each element [i,j] represents the interaction energy between
              particles i and j.
    """
    zero_pair_mask, one_pair_mask, two_pair_mask = (pair_mask == 0), (pair_mask == 1), (pair_mask == 2)

    dr.requires_grad_(True)
    # Initialize t_1, t_2, t_3, t_4
    t_1, t_2 = torch.zeros_like(dr, requires_grad=False),torch.zeros_like(dr, requires_grad=False)
    t_3, t_4 = torch.zeros_like(dr, requires_grad=False), torch.zeros_like(dr, requires_grad=False)

    # Calculate t_1: coulombic interaction term
    t_1[zero_pair_mask] = (e * ionic_charge_i)**2   / dr[zero_pair_mask]
    t_1[one_pair_mask]  = e**2 * ionic_charge_i *ionic_charge_j / dr[one_pair_mask]
    t_1[two_pair_mask]  = (e * ionic_charge_j)**2   / dr[two_pair_mask]
    # t_1 = t_1 * J_to_eV      # convert back to eV
    t_1.requires_grad_(True)

    # Calculate t_2: the Born-Huggins exponential repulsion term
    t_2[zero_pair_mask] = a[0] * torch.exp(b*(sigma_ij[0] - dr[zero_pair_mask]))
    t_2[one_pair_mask]  = a[1] * torch.exp(b*(sigma_ij[1] - dr[one_pair_mask]))
    t_2[two_pair_mask]  = a[2] * torch.exp(b*(sigma_ij[2] - dr[two_pair_mask]))
    t_2.requires_grad_(True)
    # Calculate t_3: the dipole-dipole dispersion energies term
    t_3[zero_pair_mask] = c[0] / torch.pow(dr[zero_pair_mask], 6)
    t_3[one_pair_mask]  = c[1] / torch.pow(dr[one_pair_mask], 6)
    t_3[two_pair_mask]  = c[2] / torch.pow(dr[two_pair_mask], 6)
    t_3.requires_grad_(True)

    # Calculate t_4: the dipole-quadrupole dispersion energies term
    t_4[zero_pair_mask] = d[0] / torch.pow(dr[zero_pair_mask], 8)
    t_4[one_pair_mask]  = d[1] / torch.pow(dr[one_pair_mask], 8)
    t_4[two_pair_mask]  = d[2] / torch.pow(dr[two_pair_mask], 8)
    t_4.requires_grad_(True)

    # Calculate the potential energy
    potential_energy = t_1 + t_2 - t_3 - t_4

    # Handle potential numerical instabilities and infinities
    return torch.where(dr>0, potential_energy, torch.zeros_like(potential_energy))

def tomi_fusi_pair_force(
        dr: torch.Tensor,
        potential_energy: torch.Tensor,
) -> torch.Tensor:
    """Calculate pairwise Fumi-Tosi forces between particles.

    Uses torch.autograd.grad() to calculate the force.

    Args:
        dr: Pairwise distances between particles. shape: [n,m]
        potential_energy: Pairwise Fumi-Tosi potential energies between particles.

    Returns:
        torch.Tensor: Pairwise Fumi-Tosi forces between particles. Shape: [n,m]
            Each element [i,j] represents the force magnitude between particles i and j.
            Positive values indicate repulsion and negative values indicate attraction.
    """
    # Initialize force tensor
    force_init = -1 * torch.ones_like(potential_energy)

    # Calculate the force (negative gradient of potential)
    force = torch.autograd.grad(
        outputs=potential_energy,
        inputs=dr,
        grad_outputs=force_init,
        retain_graph=False,
        create_graph=False,
    )[0]

    return torch.where(dr>0, force, torch.zeros_like(force))

class FumiTosiModel(ModelInterface):
    """Fumi-Tosi potential energy and force calculator.

    Implements the Fumi-Tosi potential for molecular dynamics simulations.
    This model calculates pairwise interactions between atoms and supports either
    full pairwsie calculation or neighbor list-based optimization for efficiency.

    Attributes:
        atomic_number_zi (int): Atomic number of species i (periodic table integer, e.g. Li=3, F=9)
        ionic_charge_i (int): Ionic charge of species i (e.g. +1 for Li+)
        ionic_charge_j (int): Ionic charge of species i (e.g. -1 for F-)
        sigma_ij (torch.Tensor): Born-Mayer-Huggins repulsive parameter (Interaction-dependent length parameter)
        a (torch.Tensor): Aij parameter (Born-Mayer-Huggins-Tosi-Fumi repulsive parameter)
        b (float): B parameter (B = 1/rho) (Born-Mayer-Huggings-Tosi-Fumi repulsive parameter)
        c (torch.Tensor): Cij parameter (Van der Waals parameter)
        d (torch.Tensor): Dij parameter (Van der Waals parameter)
        rc (float): radial cutoff
        device (torch.device): Device used for computation/calculations.
        dtype (torch.dtype): Data type used for computations/calculations.
        compute_forces (bool): Whether to compute atomic forces.
        compute_stress (bool): Whether to compute stress tensor.
        per_atom_energies (bool) Whether to compute per-atom potential energy decomposition.
        per_atom_stresses (bool) Whether to compute per-atom stress decomposition.
        use_neighbor_list (bool): Whether to use neighbor optimization.
    """

    def __init__(
            self,
            atomic_number_zi: int,
            ionic_charge_i: int,
            ionic_charge_j: int,
            sigma_ij: list | torch.Tensor,
            a: list | torch.Tensor,
            b: float | torch.Tensor,
            c: list | torch.Tensor,
            d: list | torch.Tensor,
            rc: float = DEFAULT_Rc,
            device: torch.device = None,
            dtype: torch.dtype = torch.float32,
            *, # Force keyword-only arguments
            compute_forces: bool = True,
            compute_stress: bool = False,
            per_atom_energies: bool = True,
            per_atom_stresses: bool = False,
            use_neighbor_list: bool = True,
    ) -> None:

        """Initialize the Fumi-Tosi potential energy and force calculator.

        Creates a modle with specified interaction parameters and computational flags.
        The model can be configured to compute different properties (forces, stresses)
        and use different optimization strategies.

        Args:
            atomic_number_zi (int): Atomic number of species i (periodic table integer, e.g. Li=3, F=9)
            ionic_charge_i (int): Ionic charge of species i (e.g. +1 for Li+)
            ionic_charge_j (int): Ionic charge of species i (e.g. -1 for F-)
            sigma_ij (list | torch.Tensor): Born-Mayer-Huggins repulsive parameter (Interaction-dependent length parameter)
            a (list | torch.Tensor): Aij parameter (Born-Mayer-Huggins-Tosi-Fumi repulsive parameter)
            b (float | torch.Tensor): B parameter (B = 1/rho) (Born-Mayer-Huggings-Tosi-Fumi repulsive parameter)
            c (list | torch.Tensor): Cij parameter (Van der Waals parameter)
            d (list | torch.Tensor): Dij parameter (Van der Waals parameter)
            rc (float): radial cutoff (defaults to 7 Angs)
            device (torch.device): Device used for computation/calculations.
            dtype (torch.dtype): Data type used for computations/calculations.
            compute_forces (bool): Whether to compute atomic forces. (true by default)
            compute_stress (bool): Whether to compute stress tensor.
            per_atom_energies (bool) Whether to compute per-atom potential energy decomposition. (true by default)
            per_atom_stresses (bool) Whether to compute per-atom stress decomposition.
            use_neighbor_list (bool): Whether to use neighbor optimization. (true by default)

        Example:
            a = torch.tensor([0.421999990940094, 0.29010000824928284, 0.1581999957561493])
            c = torch.tensor([0.045570001006126404, 1.2484999895095825, 69.29000091552734])
            d = torch.tensor([0.018727000802755356, 1.4982000589370728, 139.2100067138672])
            licl_model = FumiTosiModel(
                atomic_number_zi=3,                                # Atomic number of Li
                ionic_charge_i= 1,                                 # ionic charge of Li
                ionic_charge_j= -1,                                # ionic charge of Cl
                sigma_ij= torch.tensor([1.632, 2.401, 3.170]),     # sigma_ij (in Å) for Li-Li, Li-Cl & Cl-Cl pairs
                a=a,                                               # Aij (in eV)
                b=torch.tensor([2.9200]),                          # B (in Å^(-1))
                c=c,                                               # Cij (in eV * Å^(6))
                d=d,                                               # Dij (in eV * Å^8)
                rc=7,                                              # radial cutoff (in Å)
                device=torch.device('cuda'),
                dtype=torch.float32,                               # defaults to torch.float32
                compute_forces=True,                               # True by default
                compute_stress=True,
                per_atom_energies=True,                            # True by default
                per_atom_stresses=True,
                use_neighbor_list=True,                            # True by default
            )
        """
        super().__init__()
        self._device = device or torch.device('cpu')
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self.per_atom_energies = per_atom_energies
        self.per_atom_stresses = per_atom_stresses
        self.use_neighbor_list = use_neighbor_list

        # Convert parameters to tensors
        self.atomic_number_zi = atomic_number_zi
        self.ionic_charge_i = torch.tensor(ionic_charge_i, dtype=dtype, device=self._device)
        self.ionic_charge_j = torch.tensor(ionic_charge_j, dtype=dtype, device=self._device)
        self.sigma_ij = sigma_ij if isinstance(sigma_ij, torch.Tensor) else torch.tensor(sigma_ij, dtype=dtype, device=self._device)
        self.a = a if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=dtype, device=self._device)
        self.b = b if isinstance(b, torch.Tensor) else torch.tensor(b, dtype=dtype, device=self._device)
        self.c = c if isinstance(c, torch.Tensor) else torch.tensor(c, dtype=dtype, device=self._device)
        self.d = d if isinstance(d, torch.Tensor) else torch.tensor(d, dtype=dtype, device=self._device)
        self.rc = torch.tensor(rc, dtype=dtype, device=self._device)

    def unbatched_forward(
            self,
            state: SimState,
    ) -> dict[str, torch.Tensor]:
        """Compute Fumi-Tosi properties for a single unbatched state.

        Internal implementation that processes a single, non-batched simulation state.
        This method handles the core computations fo pair interactions, neighbor lists,
        and property calculations.

        Args:
            state (SimState): Single, non-batched simulation state containing atomic
                positions, cell vectors, and other system information.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing Computed Fumi-Tosi properties
                "energy": Total potential energy (scalar)
                "energies": Per-atom potential energy with shape [n_atoms](if
                per_atom_energies=True)
                "forces": Atomic forces with shape [n_atoms, 3] (if compute_forces=True)
                "stress": Stress tensor with shape [3, 3] (if compute_stress=True)
                "stresses": Per-atom stresses with shape [n_atoms, 3, 3] (if
                per_atom_stresses=True)

        Notes:
            This method handles two different approaches:
            1. Neighbor list approach: Efficient for larger systems.
            2. Full pairwise calculation: Better for small systems.

            The implementation applies cutoff distance to both approaches for consistency
        """
        if not isinstance(state, SimState):
            state = SimState(**state)

        positions = state.positions
        cell = state.row_vector_cell
        cell = cell.squeeze()
        pbc  = state.pbc

        # Ensure system_idx ecists (create if None for single system)
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
            mapping, system_mapping, shifts_idx = torchsim_nl(
                positions= wrapped_positions,
                cell= cell,
                pbc = pbc,
                cutoff = self.rc,
                system_idx= system_idx,
            )
            pairs = (mapping[0], mapping[1])
            # Pass shifts_idx directly - get_pair_displacements will convert them
            dr_vec, distances = transforms.get_pair_displacements(
                positions = wrapped_positions,
                cell = cell,
                pbc = pbc,
                pairs = pairs,
                shifts = shifts_idx
            )
            # Map atomic species: zi to 0 and zj to 1
            types = (state.atomic_numbers != self.atomic_number_zi).int() # (N,) values 0 or 1
            # pairs: tuple of index tensors i,j both shape (M,) M = N*(N-1)/2
            i_idx, j_idx = pairs
            type_i = types[i_idx]
            type_j = types[j_idx]
            # unordered pair index mapping for two species
            # pair_mask values: 0 -> (zi,zi) 1-> (zi,zj) or (zj,zi), 2 -> (zj,zj)
            pair_mask = torch.minimum(type_i, type_j) + torch.maximum(type_i, type_j)



        else:
            # Get all pairwise displacements
            dr_vec, distances = transforms.get_pair_displacements(
                positions = wrapped_positions, cell = cell, pbc = pbc
            )
            # Mass out self-interactions
            mask = torch.eye(
                wrapped_positions.shape[0], dtype=torch.bool, device=self.device
            )
            distances = distances.masked_fill(mask, float("inf"))

            # Apply cutoff
            mask = distances < self.rc
            # Get valid pairs - match neighbor list convention for pair order
            i,j = torch.where(mask)
            mapping = torch.stack([j,i])
            pairs = (mapping[0], mapping[1])
            # Map atomic species: zi to 0 and zj to 1
            types = (state.atomic_numbers != self.atomic_number_zi).int() # (N,) values 0 or 1
            # pairs: tuple of index tensors i,j both shape (M,) M = N*(N-1)/2
            i_idx, j_idx = pairs
            type_i = types[i_idx]
            type_j = types[j_idx]
            # unordered pair index mapping for two species
            # pair_mask values: 0 -> (zi,zi) 1-> (zi,zj) or (zj,zi), 2 -> (zj,zj)
            pair_mask = torch.minimum(type_i, type_j) + torch.maximum(type_i, type_j)

            # Get valid displacements and distances
            dr_vec = dr_vec[mask]
            distances = distances[mask]

        # Calculate pair energies and apply cutoff
        pair_energies = tomi_fusi_pair(
            dr=distances, pair_mask=pair_mask,
            ionic_charge_i=self.ionic_charge_i, ionic_charge_j=self.ionic_charge_j,
            sigma_ij=self.sigma_ij,
            a=self.a, b=self.b, c=self.c, d=self.d,
        )

        # Zero out energies beyond cutoff
        mask = distances < self.rc
        pair_energies = torch.where(mask, pair_energies, torch.zeros_like(pair_energies))

        # Initialize results with total energy (sum/2 to avoid double counting)
        results = {'energy': 0.5 * pair_energies.detach().sum()}

        if self.per_atom_energies:
            atom_energies = torch.zeros(
                positions.shape[0], dtype=self.dtype, device=self.device
            )
            # Each atom gets half of the pair energy
            atom_energies.index_add_(0, mapping[0], 0.5 * pair_energies)
            atom_energies.index_add_(0, mapping[1], 0.5 * pair_energies)

            results["energies"] = atom_energies.detach()

        if self.compute_forces or self.compute_stress:
            # Calculate forces and apply cutoff
            pair_forces = tomi_fusi_pair_force(
                dr=distances,potential_energy=pair_energies
            )
            pair_forces = torch.where(mask, pair_forces, torch.zeros_like(pair_forces))

            # Project forces along displacement vectors
            force_vectors = (pair_forces/ distances)[:, None] * dr_vec

            if self.compute_forces:
                # Initialize forces tensor
                forces = torch.zeros_like(positions)
                # Add force contributions (f_ij on i, -f_ij on j)
                forces.index_add_(0, mapping[0], -force_vectors)
                forces.index_add_(0, mapping[1], force_vectors)
                results["forces"] = forces.detach()

            if self.compute_stress and cell is not None:
                # Compute stress tensor
                stress_per_pair = torch.einsum("...i,...j->...ij", dr_vec, force_vectors)
                volume = torch.abs(torch.linalg.det(cell))
                results["stresses"] = -stress_per_pair.detach().sum(dim=0) / volume

                if self.per_atom_stresses:
                    atom_stresses = torch.zeros(
                        (state.positions.shape[0], 3,3), dtype=self.dtype, device=self.device,
                    )
                    atom_stresses.index_add_(0, mapping[0], -0.5 * stress_per_pair)
                    atom_stresses.index_add_(0, mapping[1], -0.5 * stress_per_pair)
                    results["stresses"] = atom_stresses.detach() / volume

        return results

    def forward(self, state: SimState | StateDict) -> dict[str, torch.Tensor]:
        """Compute Fumi-Tosi properties (energies, forces, and stresses) for a system.

        Main entry point for Fumi-Tosi calculations that handles batched states by
        dispatching each system to the unbatched implementation and combining results.

        Args:
            state (SimState | StateDict): Input state containing atomic positions,
                cell vectors, and other system information. Can be a SimState object
                or a dictionary with the same keys.

        Returns:
            dict[str, torch.Tensor]: Computed properties:
                "energy": Total potential energy (scalar)
                "energies": Per-atom potential energy with shape [n_atoms](if
                    per_atom_energies=True)
                "forces": Atomic forces with shape [n_atoms, 3] (if compute_forces=True)
                "stress": Stress tensor with shape [3, 3] (if compute_stress=True)
                "stresses": Per-atom stresses with shape [n_atoms, 3, 3] (if
                    per_atom_stresses=True)
        Raises:
            ValueError: If system cannot be inferred for multi-cell systems.
        """
        sim_state = (
            state
            if isinstance(state, SimState)
            else SimState(**state, masses=torch.ones_like(state["positions"]))
        )

        if sim_state.system_idx is None and sim_state.cell.shape[0] > 1:
            raise ValueError("System can only be inferred for batch size 1.")

        outputs = [
            self.unbatched_forward(sim_state[idx]) for idx in range(sim_state.n_systems)
        ]
        properties = outputs[0]

        # Note: Returned properties are always torch.Tensor
        results: dict[str, torch.Tensor] = {}

        for key in ("stress", "energy"):
            if key in properties:
                results[key] = torch.stack([out[key] for out in outputs])

        for key in ("energies", "forces", "stresses"):
            if key in properties:
                results[key] = torch.cat([out[key] for out in outputs])

        return results