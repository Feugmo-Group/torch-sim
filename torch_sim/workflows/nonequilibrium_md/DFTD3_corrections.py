import os
from pathlib import Path
import tarfile
from hashlib import md5
import numpy as np
import requests
import re
import io

import torch
from nvalchemiops.torch.interactions.dispersion import D3Parameters, dftd3
from nvalchemiops.neighbors import neighbor_list

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
            save_params: bool=True,
    ):
        """
        For List of functionals and coefficients for BJ-damping refer to: (Table: s6 a1 s8 a2)
        https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/bj_damping
        """
        # Loading the dftd3 parameters from the parameter file
        self.dftd3_parameters = self.load_dftd3_params(parameter_file, save_params)

        # Maximum radius for calculation the dftd3 dispersion (in Bohr)
        self.r_max = r_max * ANGSTROM_TO_BOHR # neighbor list expects cutoff radius in bohr
        self.a1 = a1                          # BJ damping parameter
        self.a2 = a2                          # BJ damping radius
        self.s6 = s6                          # C6 term coefficient
        self.s8 = s8                            # C8 term coefficient


    def compute_corrections(
            self,
            state_positions: torch.Tensor,
            state_cell: torch.Tensor,
            system_idx: torch.Tensor,
            state_atomic_numbers: torch.Tensor,
            state_pbc: torch.Tensor,
    )-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        state_atomic_numbers = state_atomic_numbers.to(torch.int32)
        system_idx = system_idx.to(torch.int32)

        # Generate the neighbor list based on r_max
        mapping, neigh_ptr, shifts_idx = neighbor_list(
            positions=state_positions,
            cutoff=self.r_max,
            cell=state_cell,
            pbc=state_pbc,
            return_neighbor_list=True,
            batch_idx=system_idx,
        )

        # Compute dftd3 corrections using nvidia alchemi toolkit ops
        energy, forces, coord_num, virial = dftd3(
            state_positions * ANGSTROM_TO_BOHR,
            state_atomic_numbers,
            a1=self.a1,  # BJ damping parameter (PBE0)
            a2=self.a2,  # BJ damping radius (PBE0)
            s6=self.s6,
            s8=self.s8,  # C8 term coefficient (PBE0)
            d3_params=self.dftd3_parameters,
            neighbor_list=mapping,
            neighbor_ptr=neigh_ptr,
            cell=state_cell * ANGSTROM_TO_BOHR,
            unit_shifts=shifts_idx,
            compute_virial=True,
            batch_idx=system_idx,
            )
        # Convert the units:
        energy = energy * HARTREE_TO_EV                          # Energy: Hartree -> eV
        forces = forces * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)     # Forces: Hartree/Bohr -> eV/Angstrom
        virial = virial * (HARTREE_TO_EV / BOHR_TO_ANGSTROM**3)  # Virial: Hartree/Bohr^3 -> eV/Ang^3
        return energy, forces, virial

    @staticmethod
    def load_dftd3_params(parameter_file: Path=None, save_params: bool=True):

        # Check if parameter_file ends in .pt:

        # If a path is provided and corresponds to a .pt file, load the parameters directly.
        if isinstance(parameter_file, Path) and parameter_file.is_file() and parameter_file.suffix == ".pt":
            params = torch.load(parameter_file, weights_only=True)

        # If no path is provided, check if default cached parameters exist in the current working directory and load them.
        elif parameter_file is None and os.path.exists(os.path.join(os.getcwd(), "dftd3_params")) and os.path.exists(os.path.join(os.getcwd(), "dftd3_params", "dftd3_parameters.pt")):
            params = torch.load(os.path.join(os.getcwd(), "dftd3_params", "dftd3_parameters.pt"), weights_only=True)

        # If a path is provided but it is not a .pt file, assume it points to a directory containing source files (.f) and generate parameters from it.
        elif isinstance(parameter_file, Path) and not parameter_file.is_file() and parameter_file.suffix != ".pt":
            params = extract_dftd3_parameters(parameter_file)

        # If no path provided and defaults are missing, generate parameters by downloading source files.
        else:
            params = extract_dftd3_parameters(parameter_file)

        # Save Params:
        if save_params:
            params_folder = Path(os.getcwd()) / "dftd3_params"
            params_folder.mkdir(parents=True, exist_ok=True)
            params_file = params_folder / "dftd3_parameters.pt"
            torch.save(params, params_file)
            print(f"Saved parameters to: {params_file}")
        return params




# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES
# Helper functions and variables from NVIDIA/nalchemi-toolkit-ops (Apache 2.0)
# Original source: https://github.com/NVIDIA/nalchemi-toolkit-ops/blob/main/examples/dispersion/utils.py
# URL for DFT-D3 parameter files from Grimme group
DFTD3_TGZ_URL = "https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/dftd3.tgz"
REFERENCE_MD5 = "a76c752e587422c239c99109547516d2"

def extract_dftd3_parameters(
    dftd3_ref_dir: Path | None = None,
) -> dict[str, torch.Tensor]:
    """
    Extract DFT-D3 parameters from the reference data.

    This function either reads local Fortran source files or downloads them from
    the Grimme group website, then extracts the DFT-D3 parameters needed for
    dispersion corrections. The parameters are then converted to a dictionary
    of PyTorch tensors in the expected format for the ``nvalchemiops`` DFT-D3 kernels.

    This method is intended for convenience, and it is possible for users to
    provide alternative DFT-D3 parameters provided they are in the expected
    format.

    Parameters
    ----------
    dftd3_ref_dir : Path or None, optional
        Path to directory containing dftd3.f and pars.f from the reference
        implementation. If None, the reference data is pulled from the
        Grimme group page.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing parameter tensors:
        - "rcov": Covalent radii [95] in Bohr (float32)
        - "r4r2": <r⁴>/<r²> expectation values [95] (float32)
        - "c6ab": C6 reference values [95, 95, 5, 5] (float32)
        - "cn_ref": CN reference grid [95, 95, 5, 5] (float32)

    Raises
    ------
    FileNotFoundError
        If dftd3_ref_dir is provided but files don't exist
    requests.RequestException
        If download fails (when dftd3_ref_dir is None)
    ValueError
        If parsing fails

    Notes
    -----
    When downloading (dftd3_ref_dir=None), files are fetched from:
    - https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/dftd3.tgz

    The archive is downloaded and extracted in-memory to obtain dftd3.f and pars.f.

    All parameters are in atomic units (Bohr for distances).
    Index 0 is reserved for padding; valid atomic numbers are 1-94.

    Examples
    --------
    Download parameters automatically:

    >>> params = generate_dftd3_parameters()
    >>> params["rcov"].shape
    torch.Size([95])

    Use local files:

    >>> params = generate_dftd3_parameters(Path("path/to/dftd3_ref"))
    >>> params["rcov"].shape
    torch.Size([95])
    """
    if dftd3_ref_dir is not None:
        # Use local files
        if not dftd3_ref_dir.exists():
            raise FileNotFoundError(f"Directory not found: {dftd3_ref_dir}")

        dftd3_f = dftd3_ref_dir / "dftd3.f"  # NOSONAR
        pars_f = dftd3_ref_dir / "pars.f"  # NOSONAR

        if not dftd3_f.exists():
            raise FileNotFoundError(f"File not found: {dftd3_f}")
        if not pars_f.exists():
            raise FileNotFoundError(f"File not found: {pars_f}")

        print(f"Reading DFT-D3 parameter files from: {dftd3_ref_dir}")
        with open(dftd3_f) as f:
            dftd3_content = f.read()
        with open(pars_f) as f:
            pars_content = f.read()
        print("  ✓ Files loaded")
    else:
        # Download from web
        print("Downloading DFT-D3 parameter files from Grimme group website...")
        try:
            print(f"  Downloading archive from {DFTD3_TGZ_URL}")
            extracted_files = _download_and_extract_tgz(DFTD3_TGZ_URL)
            print("  ✓ Download and extraction complete")

            # Extract the required files
            if "dftd3.f" not in extracted_files:
                raise ValueError("dftd3.f not found in archive")
            if "pars.f" not in extracted_files:
                raise ValueError("pars.f not found in archive")

            dftd3_content = extracted_files["dftd3.f"]
            pars_content = extracted_files["pars.f"]

        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to download DFT-D3 parameter files: {e}\n"
                "Please check your internet connection and try again, or provide "
                "a local dftd3_ref directory path."
            ) from e
        except (tarfile.TarError, ValueError) as e:
            raise RuntimeError(
                f"Failed to extract DFT-D3 parameter files from archive: {e}\n"
                "The archive format may have changed. Please provide "
                "a local dftd3_ref directory path."
            ) from e

    # Parse Fortran arrays
    print("Parsing Fortran source files...")
    r2r4_94 = _find_fortran_array(dftd3_content, "r2r4")
    rcov_94 = _find_fortran_array(dftd3_content, "rcov")
    pars_records = _parse_pars_array(pars_content)

    # Build parameter arrays (index 0 reserved, elements 1-94)
    r4r2 = np.zeros(95, dtype=np.float32)
    r4r2[1:95] = r2r4_94.astype(np.float32)

    rcov = np.zeros(95, dtype=np.float32)
    rcov[1:95] = rcov_94.astype(np.float32)

    c6ab, cn_ref = _build_arrays(pars_records)
    print("  ✓ Parsing complete")

    # Convert to PyTorch tensors
    return {
        "rcov": torch.from_numpy(rcov),
        "r4r2": torch.from_numpy(r4r2),
        "c6ab": torch.from_numpy(c6ab),
        "cn_ref": torch.from_numpy(cn_ref),
    }

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES
# Helper functions and variables from NVIDIA/nalchemi-toolkit-ops (Apache 2.0)
# Original source: https://github.com/NVIDIA/nalchemi-toolkit-ops/blob/main/examples/dispersion/utils.py
def _download_and_extract_tgz(url: str) -> dict[str, str]:
    """
    Download and extract Fortran source files from .tgz archive.

    Parameters
    ----------
    url : str
        URL to .tgz archive

    Returns
    -------
    dict[str, str]
        Dictionary mapping filenames to their contents.
        Keys are base filenames (e.g., "dftd3.f", "pars.f")

    Raises
    ------
    requests.RequestException
        If download fails
    tarfile.TarError
        If extraction fails
    ValueError
        If MD5 checksum verification fails
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Verify MD5 checksum
    content_bytes = response.content
    hasher = md5(usedforsecurity=False)
    hasher.update(content_bytes)
    computed_md5 = hasher.hexdigest()
    if computed_md5 != REFERENCE_MD5:
        raise ValueError(
            f"MD5 checksum verification failed for downloaded archive.\n"
            f"Expected: {REFERENCE_MD5}\n"
            f"Got:      {computed_md5}\n"
            "The archive may have been modified or corrupted. "
            "Please verify the source or provide a local dftd3_ref directory path."
        )

    # Extract files from tar.gz archive
    extracted_files = {}
    with tarfile.open(fileobj=io.BytesIO(content_bytes), mode="r:gz") as tar:  # NOSONAR
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith((".f", ".F")):
                # Extract file content
                file_obj = tar.extractfile(member)
                if file_obj is not None:
                    content = file_obj.read().decode("utf-8", errors="ignore")
                    # Store with base filename as key
                    basename = Path(member.name).name
                    extracted_files[basename] = content

    return extracted_files


# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES
# Helper functions and variables from NVIDIA/nalchemi-toolkit-ops (Apache 2.0)
# Original source: https://github.com/NVIDIA/nalchemi-toolkit-ops/blob/main/examples/dispersion/utils.py
def _find_fortran_array(content: str, var_name: str) -> np.ndarray:
    """
    Parse Fortran data array by variable name, skipping comments.

    Parameters
    ----------
    content : str
        Fortran source file content
    var_name : str
        Variable name to search for

    Returns
    -------
    np.ndarray
        Parsed array values as float64

    Raises
    ------
    ValueError
        If variable not found or parsing fails
    """
    lines = content.splitlines()

    in_data_block = False
    data_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("!") or stripped.lower().startswith("c "):
            continue

        if not in_data_block:
            if re.match(rf"^\s*data\s+{var_name}\s*/\s*", line, re.IGNORECASE):
                in_data_block = True
                data_lines.append(line)
        else:
            data_lines.append(line)
            if "/" in line and not line.strip().startswith("!"):
                break

    if not data_lines:
        raise ValueError(f"Variable '{var_name}' not found in Fortran source")

    data_str = " ".join(data_lines)
    pattern = rf"data\s+{var_name}\s*/\s*(.*?)\s*/"
    match = re.search(pattern, data_str, re.DOTALL | re.IGNORECASE)

    if not match:
        raise ValueError(f"Failed to parse '{var_name}'")

    content = match.group(1)
    lines_clean = []
    for line in content.split("\n"):
        if "!" in line:
            line = line[: line.index("!")]
        lines_clean.append(line)
    content = " ".join(lines_clean)

    numbers = re.findall(r"[-+]?\d+\.\d+(?:_wp)?", content)  # NOSONAR
    values = [float(n.replace("_wp", "")) for n in numbers]

    return np.array(values, dtype=np.float64)


# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES
# Helper functions and variables from NVIDIA/nalchemi-toolkit-ops (Apache 2.0)
# Original source: https://github.com/NVIDIA/nalchemi-toolkit-ops/blob/main/examples/dispersion/utils.py
def _parse_pars_array(content: str) -> np.ndarray:
    """
    Parse pars array containing [C6, Z_i, Z_j, CN_i, CN_j] records.

    Parameters
    ----------
    content : str
        Fortran pars.f file content

    Returns
    -------
    np.ndarray
        Array of shape [n_records, 5] containing parameter records

    Notes
    -----
    Each record contains:
    - C6 coefficient value
    - Encoded atomic number for element i
    - Encoded atomic number for element j
    - Coordination number for element i
    - Coordination number for element j
    """
    values = []
    in_data_section = False

    for line in content.splitlines():
        if "real*8" in line.lower() and "pars" in line.lower():
            continue

        if "pars(" in line.lower() and "=(" in line:
            in_data_section = True

        if not in_data_section:
            continue

        if "/)" in line:
            in_data_section = False

        if "!" in line:
            line = line[: line.index("!")]

        line = line.replace("pars(", " ").replace("=(/", " ")
        line = line.replace("/)", " ").replace(":", " ")

        numbers = re.findall(r"[-+]?\d+\.\d+[eEdD][-+]?\d+", line)  # NOSONAR
        values.extend(
            float(num_str.replace("D", "e").replace("d", "e")) for num_str in numbers
        )

    values = np.array(values, dtype=np.float64)
    n_records = len(values) // 5

    if len(values) % 5 != 0:
        values = values[: n_records * 5]

    return values.reshape(n_records, 5)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES
# Helper functions and variables from NVIDIA/nalchemi-toolkit-ops (Apache 2.0)
# Original source: https://github.com/NVIDIA/nalchemi-toolkit-ops/blob/main/examples/dispersion/utils.py
def _limit(encoded: int) -> tuple[int, int]:
    """
    Decode Fortran element encoding.

    The Fortran implementation encodes atomic number and coordination number
    index into a single integer.

    Parameters
    ----------
    encoded : int
        Encoded value from Fortran (atomic_number + 100 * (cn_index - 1))

    Returns
    -------
    atom : int
        Atomic number (1-94)
    cn_idx : int
        Coordination number index (1-5)

    Examples
    --------
    >>> _limit(101)
    (1, 2)
    >>> _limit(201)
    (1, 3)
    """
    atom = encoded
    cn_idx = 1

    while atom > 100:
        atom -= 100
        cn_idx += 1

    return atom, cn_idx

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES
# Helper functions and variables from NVIDIA/nalchemi-toolkit-ops (Apache 2.0)
# Original source: https://github.com/NVIDIA/nalchemi-toolkit-ops/blob/main/examples/dispersion/utils.py
def _build_arrays(pars_records: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build c6ab and cn_ref arrays from pars records.

    Parameters
    ----------
    pars_records : np.ndarray
        Array of shape [n_records, 5] from _parse_pars_array

    Returns
    -------
    c6ab : np.ndarray
        C6 coefficients array [95, 95, 5, 5] as float32
    cn_ref : np.ndarray
        Coordination number reference grid [95, 95, 5, 5] as float32

    Notes
    -----
    Arrays are indexed from 0 but element 0 is unused (reserved for padding).
    Valid atomic numbers are 1-94.
    """
    c6ab = np.zeros((95, 95, 5, 5), dtype=np.float32)
    cn_ref = np.full((95, 95, 5, 5), -1.0, dtype=np.float32)
    cn_values = {elem: {} for elem in range(95)}

    for record in pars_records:
        c6_val, z_i_enc, z_j_enc, cn_i, cn_j = record

        iat, iadr = _limit(int(z_i_enc))
        jat, jadr = _limit(int(z_j_enc))

        if iat < 1 or iat > 94 or jat < 1 or jat > 94:
            continue
        if iadr < 1 or iadr > 5 or jadr < 1 or jadr > 5:
            continue

        iadr_py = iadr - 1
        jadr_py = jadr - 1

        c6ab[iat, jat, iadr_py, jadr_py] = c6_val
        c6ab[jat, iat, jadr_py, iadr_py] = c6_val

        if iadr_py not in cn_values[iat]:
            cn_values[iat][iadr_py] = cn_i
        if jadr_py not in cn_values[jat]:
            cn_values[jat][jadr_py] = cn_j

    for elem in range(1, 95):
        for partner in range(1, 95):
            for cn_idx in range(5):
                if cn_idx in cn_values[elem]:
                    cn_ref[elem, partner, cn_idx, :] = cn_values[elem][cn_idx]

    return c6ab, cn_ref

