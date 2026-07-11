import os
from pathlib import Path
import torch
from nvalchemiops.torch.interactions.dispersion._dftd3 import D3Parameters, dftd3
from nvalchemiops.neighbors import neighbor_list
from nvalchemiops.torch.interactions.dispersion._dftd3 import D3Parameters
# Import utilities from nvalchemi-toolkit-ops repo's utils.py (examples/dispersion/utils.py)
from utils import (
    DFTD3,
    extract_dftd3_parameters,
    save_dftd3_parameters,
    load_d3_parameters,
)

# Unit conversion constants from CODATA 2022 (retrieved 2025-11-12)
# Bohr radius: 5.291 772 105 44 x 10^-11 m
# Hartree energy in eV: 27.211 386 245 981 eV
BOHR_TO_ANGSTROM = 0.529177210544
HARTREE_TO_EV = 27.211386245981
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

class DFTD3Corrections():
    """DFT-D3 corrections based on nvidia's alchemiops toolkit"""
    def __init__(
            self,
            parameter_file=None,
            r_max: float=50,
            a1: float=0.4145,
            a2: float=4.8593,
            s6: float=1.0,
            s8: float=1.2177,
    ):
        """
        For List of functionals and coefficients for BJ-damping refer to: (Table: s6 a1 s8 a2)
        https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/bj_damping
        """
        # Loading the dftd3 parameters from the parameter file
        if parameter_file is not None:
            self.dftd3_parameters = load_d3_parameters(parameter_file)
        else:
            self.dftd3_parameters = self.load_parameters_from_file(parameter_file)

        # Maximum radius for calculation the dftd3 dispersion (in Angs)
        self.r_max = r_max 
        self.a1=a1  # BJ damping parameter
        self.a2=a2  # BJ damping radius
        self.s6=s6  # C6 term coefficient
        self.s8=s8  # C8 term coefficient

    def load_parameters_from_file(self, parameter_file):
        # Code from Example Gallery
        # Link: https://nvidia.github.io/nvalchemi-toolkit-ops/examples/dispersion/01_dftd3_molecule.html

        # Check for cached parameters, download if needed
        # This step downloads ~500 KB of reference data from the Grimme group
        param_file = (
                Path(os.path.expanduser("~")) / ".cache" / "nvalchemiops" / "dftd3_parameters.pt"
        )
        if not param_file.exists():
            # Possible license problem!
            print("Downloading DFT-D3 parameters...")
            params = extract_dftd3_parameters()
            save_dftd3_parameters(params)
        else:
            params = torch.load(param_file, weights_only=True)
            print("Loaded cached DFT-D3 parameters")
        return params

    def compute_corrections(
            self,
            state_positions: torch.Tensor,
            state_cell: torch.Tensor,
            system_idx: torch.Tensor,
            unit_shifts: torch.Tensor,
            state_atomic_numbers: torch.Tensor
    )-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        state_atomic_numbers = state_atomic_numbers.to(torch.int32)

        # Generate the neighbor list based on r_max
        mapping, neigh_ptr, shifts_idx = neighbor_list(
            positions=state_positions,
            cutoff=self.r_max,
            cell=state_cell,
            pbc=True,
            return_neighbor_list=True,
            batch_idx=system_idx,
        )

        # Compute dftd3 corrections using nvidia alchemi toolkit ops
        energy, forces, coord_num, virial = dftd3(
            state_positions * ANGSTROM_TO_BOHR,
            state_atomic_numbers.to(torch.int32),
            a1=self.a1,  # BJ damping parameter (PBE0)
            a2=self.a2,  # BJ damping radius (PBE0)
            s6=self.s6,
            s8=self.s8,  # C8 term coefficient (PBE0)
            d3_params=self.dftd3_parameters,
            neighbor_list=mapping,
            neighbor_ptr=neigh_ptr,
            cell=state_cell * ANGSTROM_TO_BOHR,
            unit_shifts=unit_shifts,
            compute_virial=True,
            batch_idx=system_idx,
            )
        # Convert the units:
        energy = energy * HARTREE_TO_EV                          # Energy: Hartree -> eV
        forces = forces * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)     # Forces: Hartree/Bohr -> eV/Angstrom
        virial = virial * (HARTREE_TO_EV / BOHR_TO_ANGSTROM**3)  # Virial: Hartree/Bohr^3 -> eV/Ang^3
        return energy, forces, virial