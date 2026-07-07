"""Correlation function calculators for time series data.

Module provides efficient calculator for time correlation functions [1]_,
including both autocorrelation and cross-correlation functionality.
Leverages FFT-based methods [2]_ for performance and supports both CPU and
GPU acceleration through PyTorch.

The ``CorrelationCalculator`` class provides on-the-fly
correlation calculations during simulation runs, and a ``CircularBuffer``
utility class assists in data storage without frequent reallocations.

The ``VelocityAutoCorrelation`` class provides an interface for
computing the velocity autocorrelation functions (VACF).

The ``RadialDistributionFunction`` class accumulates on-the-fly pair-distance
histograms to compute g(r) with full PBC support.

The ``PressureAutoCorrelation`` class computes the off-diagonal pressure
tensor autocorrelation function, enabling Green-Kubo viscosity estimates.

References:
    .. [1] D. Frenkel and B. Smit, "Understanding molecular simulation: From
       algorithms to applications", Academic Press, 2002.
    .. [2] `pwtools: Phonon DOS <https://elcorto.github.io/pwtools/written/background/phonon_dos.html>`_
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from torch_sim.elastic import full_3x3_to_voigt_6_stress
from torch_sim.quantities import calc_heat_flux, compute_instantaneous_pressure_tensor
from torch_sim.state import SimState


class CircularBuffer:
    """Circular buffer for storing time series data.

    Provides a fixed-size circular buffer optimized for storing
    and retrieving time series data, with minimal memory allocation.

    Attributes:
        size: Maximum number of elements to store
        buffer: Storage for the data
        head: Current write position
        count: Number of elements currently stored
        device: Device where the buffer is stored
    """

    def __init__(self, size: int, device: torch.device | None = None) -> None:
        """Initialize a circular buffer.

        Args:
            size: Maximum number of elements to store
            device: Device for tensor storage (CPU or GPU)
        """
        self.size = size
        self.buffer: torch.Tensor | None = None
        self.head = 0
        self.count = 0
        self.device = device

    def append(self, value: torch.Tensor) -> None:
        """Append a new value to the buffer.

        Args:
            value (torch.Tensor): New tensor to store
        """
        if self.buffer is None:
            # Initialize buffer shape as first value
            shape = (self.size, *value.shape)
            self.buffer = torch.zeros(shape, device=self.device, dtype=value.dtype)

        if self.buffer is not None:
            self.buffer[self.head] = value
            self.head = (self.head + 1) % self.size
            self.count = min(self.count + 1, self.size)

    def get_array(self) -> torch.Tensor:
        """Get the current buffer contents as a tensor.

        Returns:
            torch.Tensor: Containing the buffered data in chronological order.
        """
        if self.count == 0 or self.buffer is None:
            return torch.empty(0, device=self.device)

        if self.count < self.size:
            # Filled portion only!
            return self.buffer[: self.count]

        # Chronological order
        # Avoid unnecessary copy if unwrapped
        if self.head == 0:
            return self.buffer

        return torch.cat([self.buffer[self.head :], self.buffer[: self.head]])

    @property
    def is_full(self) -> bool:
        """Check if the buffer is full."""
        return self.count == self.size


class CorrelationCalculator:
    """Efficient on-the-fly correlation function calculator.

    Manage the calculation of time correlation functions during
    simulation, with support for both autocorrelation and cross-correlation
    of arbitrary properties. It maintains a sliding window of historical data
    and performs efficient updates.

    Attributes:
        window_size: Number of steps to keep in memory
        properties: Map of property names to their calculators
        buffers: Circular buffers for storing historical data
        correlations: Current correlation results
        device: Device where calculations are performed

    Example:
    Computing correlation function in loop::

        corr_calc = CorrelationCalculator(
            window_size=100,
            properties={"velocity": lambda state: state.velocities},
        )

        for step in range(n_steps):
            state = integrator.step(state)
            # Call update at desired frequency
            if step % 10 == 0:  # Sample every 10 steps
                corr_calc.update(state)

            # Periodically retrieve correlation functions
            if step % 1000 == 0:
                acfs = corr_calc.get_auto_correlations()
                # Process or save acfs...
    """

    def __init__(
        self,
        *,
        window_size: int,
        properties: dict[str, Callable[[SimState], torch.Tensor]],
        device: torch.device,
        normalize: bool = True,
    ) -> None:
        """Initialize a correlation calculator.

        Args:
            window_size: Number of steps to keep in memory
            properties: Dictionary mapping names to functions that calculate
                       properties from a SimState
            device: Device for tensor storage and computation
            normalize: Whether to normalize correlation functions to [0,1]
        """
        self.window_size = window_size
        self.properties = properties or {}
        self.device = device
        self.normalize = normalize

        self.buffers = {
            name: CircularBuffer(window_size, device=device) for name in self.properties
        }

        self.correlations: dict[str, torch.Tensor] = {}
        self.cross_correlations: dict[tuple[str, str], torch.Tensor] = {}

    def add_property(
        self, name: str, calculator: Callable[[SimState], torch.Tensor]
    ) -> None:
        """Track a new simulation property.

        Args:
            name: Name of the property
            calculator: Function that calculates property from a SimState
        """
        if name in self.properties:
            raise ValueError(f"Property {name} already exists")

        self.properties[name] = calculator
        self.buffers[name] = CircularBuffer(self.window_size, device=self.device)

    def update(self, state: SimState) -> None:
        """Update correlation calculations with new state data.

        Args:
            state: Current simulation state
        """
        # Single pass update
        buffer_count = 0
        buffer_total = len(self.buffers)

        for name, calc in self.properties.items():
            value = calc(state)
            self.buffers[name].append(value)
            if self.buffers[name].count > 1:
                buffer_count += 1

        # Correlations if we have enough data
        if buffer_count == buffer_total and buffer_total > 0:
            self._compute_correlations()

    def _compute_correlations(self) -> None:  # noqa: C901, PLR0915
        """Compute correlation functions using FFT for efficiency."""
        # Autocorrelations
        for name, buf in self.buffers.items():
            data = buf.get_array()
            if len(data) == 0:
                continue

            original_shape = data.shape

            # Reshape to [time_steps, flattened_dim]
            if len(original_shape) > 1:
                data = data.reshape(original_shape[0], -1)

            n_dims = data.shape[1] if len(data.shape) > 1 else 1

            if n_dims > 1:
                # Pre-allocate/Precompute
                acf = torch.zeros(
                    (original_shape[0], n_dims), device=self.device, dtype=data.dtype
                )

                data_centered = data - data.mean(dim=0, keepdim=True)

                if data_centered.shape[1] <= 128:  # Batch Threshold
                    # Transpose for batch FFT (dimensions become batch)
                    data_batch = data_centered.T  # Shape: [n_dims, time_steps]

                    # Batch FFT processing
                    n_fft = 2 * data_batch.shape[1]
                    fft_batch = torch.fft.rfft(data_batch, n=n_fft)
                    power_batch = torch.square(torch.abs(fft_batch))
                    corr_batch = torch.fft.irfft(power_batch)[:, : data_batch.shape[1]]

                    corr_batch = corr_batch.T  # Shape: [time_steps, n_dims]

                    if self.normalize:
                        norms = corr_batch[0].clone()
                        mask = norms > 1e-10
                        if mask.any():
                            corr_batch[:, mask] = corr_batch[:, mask] / norms[
                                mask
                            ].unsqueeze(0)

                    acf = corr_batch.reshape(original_shape)
                else:
                    # Fallback for very high-dimensional data
                    for i in range(n_dims):
                        dim_data = data_centered[:, i]

                        # FFT: n=2*len for zero-padding
                        n_fft = 2 * len(dim_data)
                        fft = torch.fft.rfft(dim_data, n=n_fft)
                        power = torch.square(torch.abs(fft))
                        corr = torch.fft.irfft(power)[: len(dim_data)]

                        if self.normalize and corr[0] > 1e-10:
                            corr = corr / corr[0]

                        acf[:, i] = corr

                    # Reshape back to match input dimensions
                    acf = acf.reshape(original_shape)
            else:
                # Single dimension case
                dim_data = data - data.mean()

                n_fft = 2 * len(dim_data)
                fft = torch.fft.rfft(dim_data, n=n_fft)
                power = torch.square(torch.abs(fft))
                corr = torch.fft.irfft(power)[: len(dim_data)]

                if self.normalize and corr[0] > 1e-10:
                    corr = corr / corr[0]

                acf = corr

            self.correlations[name] = acf

        # Cross-correlations
        names = list(self.buffers)
        for i, name1 in enumerate(names):
            for name2 in names[i + 1 :]:
                data1 = self.buffers[name1].get_array()
                data2 = self.buffers[name2].get_array()

                if len(data1) == 0 or len(data2) == 0:
                    continue

                min_len = min(len(data1), len(data2))
                data1 = data1[:min_len]
                data2 = data2[:min_len]

                # Multidimensional data
                if len(data1.shape) > 1 or len(data2.shape) > 1:
                    # For now, simplify by taking mean across dimensions
                    if len(data1.shape) > 1:
                        # More efficient with tuple unpacking
                        non_time_dims = tuple(range(1, len(data1.shape)))
                        data1 = torch.mean(data1, dim=non_time_dims)
                    if len(data2.shape) > 1:
                        non_time_dims = tuple(range(1, len(data2.shape)))
                        data2 = torch.mean(data2, dim=non_time_dims)

                # Center data
                data1 = data1 - data1.mean()
                data2 = data2 - data2.mean()

                n_fft = 2 * min_len
                fft1 = torch.fft.rfft(data1, n=n_fft)
                fft2 = torch.fft.rfft(data2, n=n_fft)
                ccf = torch.fft.irfft(fft1 * fft2.conj())[:min_len]

                if self.normalize and torch.abs(ccf[0]) > 1e-10:
                    ccf = ccf / ccf[0]

                self.cross_correlations[(name1, name2)] = ccf

    def get_auto_correlations(self) -> dict[str, torch.Tensor]:
        """Get autocorrelation results.

        Returns:
            Dictionary mapping property names to their correlation tensors
        """
        return self.correlations

    def get_cross_correlations(self) -> dict[tuple[str, str], torch.Tensor]:
        """Get cross-correlation results.

        Returns:
            Dictionary mapping pairs of property names to their
            cross-correlation tensors
        """
        return self.cross_correlations

    def reset(self) -> None:
        """Reset all buffers and correlations."""
        self.buffers = {
            name: CircularBuffer(self.window_size, device=self.device)
            for name in self.properties
        }
        self.correlations = {}
        self.cross_correlations = {}

    def to(self, device: torch.device) -> "CorrelationCalculator":
        """Move calculator to specified device.

        Args:
            device: Target device

        Returns:
            Self, for method chaining
        """
        # Skip if already on target device
        if self.device == device:
            return self

        self.device = device

        new_buffers = {}
        for name, buf in self.buffers.items():
            new_buf = CircularBuffer(self.window_size, device=device)
            if buf.buffer is not None:
                data = buf.get_array().to(device)
                # Larger buffers use a batch
                if len(data) > 100:
                    # Balances memory transfer
                    batch_size = 20
                    for i in range(0, len(data), batch_size):
                        batch = data[i : min(i + batch_size, len(data))]
                        for j in range(len(batch)):
                            new_buf.append(batch[j])
                else:
                    for i in range(len(data)):
                        new_buf.append(data[i])
            new_buffers[name] = new_buf

        self.buffers = new_buffers

        # Move correlations
        if self.correlations:
            self.correlations = {
                name: corr.to(device) for name, corr in self.correlations.items()
            }

        if self.cross_correlations:
            self.cross_correlations = {
                names: corr.to(device) for names, corr in self.cross_correlations.items()
            }

        return self


class VelocityAutoCorrelation:
    """Calculator for velocity autocorrelation function (VACF).

    Computes VACF by averaging over atoms and dimensions, with optional
    running average across correlation windows.


    Using ``VelocityAutoCorrelation`` with
    :class:`~ts.trajectory.TrajectoryReporter`::

        # Create VACF calculator
        vacf_calc = VelocityAutoCorrelation(
            window_size=100,
            device=device,
            use_running_average=True,
        )

        # Set up trajectory reporter
        reporter = TrajectoryReporter(
            "simulation_vacf.h5",
            state_frequency=100,
            prop_calculators={10: {"vacf": vacf_calc}},
        )

    """

    def __init__(
        self,
        *,
        window_size: int,
        device: torch.device,
        use_running_average: bool = True,
        normalize: bool = True,
    ) -> None:
        """Initialize VACF calculator.

        Args:
            window_size: Number of steps in correlation window
            device: Computation device
            use_running_average: Whether to compute running average across windows
            normalize: Whether to normalize correlation functions to [0,1]
        """
        self.corr_calc = CorrelationCalculator(
            window_size=window_size,
            properties={"velocity": lambda s: s.velocities},
            device=device,
            normalize=normalize,
        )
        self.use_running_average = use_running_average
        self._window_count = 0
        self._avg = torch.zeros(window_size, device=device)

    def __call__(self, state: SimState, _: Any = None) -> torch.Tensor:
        """Update VACF with new state.

        Args:
            state: Current simulation state
            _: Unused model argument (required property calculator interface)

        Returns:
            Tensor containing average VACF
        """
        self.corr_calc.update(state)

        if self.corr_calc.buffers["velocity"].count == self.corr_calc.window_size:
            correlations = self.corr_calc.get_auto_correlations()
            # dims: (natoms, ndims)
            vacf = torch.mean(correlations["velocity"], dim=(1, 2))

            self._window_count += 1

            if self.use_running_average:
                factor = 1.0 / self._window_count
                self._avg += (vacf - self._avg) * factor
            else:
                self._avg = vacf

            self.corr_calc.reset()

        return torch.tensor([self._window_count], device=state.device)

    @property
    def vacf(self) -> torch.Tensor | None:
        """Current VACF result."""
        return self._avg


class HeatFluxAutoCorrelation:
    """Calculator for heat flux autocorrelation function (HFACF).

    Computes HFACF by averaging over atoms and dimensions, with optional
    running average across correlation windows.


    Using ``HeatFluxAutoCorrelation`` with
    :class:`~ts.trajectory.TrajectoryReporter`::

        # Create HFACF calculator
        hfacf_calc = HeatFluxAutoCorrelation(
            window_size=100,
            device=device,
            use_running_average=True,
            model=model,
        )

        # Set up trajectory reporter
        reporter = TrajectoryReporter(
            "simulation_hfacf.h5",
            state_frequency=100,
            prop_calculators={10: {"hfacf": hfacf_calc}},
        )

    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        window_size: int,
        device: torch.device,
        use_running_average: bool = True,
        normalize: bool = True,
    ) -> None:
        """Initialize HFACF calculator.

        Args:
            window_size: Number of steps in correlation window
            device: Computation device
            use_running_average: Whether to compute running average across windows
            normalize: Whether to normalize correlation functions to [0,1]
            model: Model to use for calculating heat flux
        """
        # TODO (AG): Figure out how to do it in a more efficient way
        self.model = model
        self.model.per_atom_stresses = True
        self.model.per_atom_energies = True

        self.corr_calc = CorrelationCalculator(
            window_size=window_size,
            properties={
                "heat_flux": lambda s: calc_heat_flux(
                    momenta=s.momenta,
                    masses=s.masses,
                    velocities=None,
                    energies=self.model(s)["energies"],
                    stresses=full_3x3_to_voigt_6_stress(self.model(s)["stresses"]),
                    batch=s.system_idx,
                    is_centroid_stress=False,
                    is_virial_only=False,
                )
            },
            device=device,
            normalize=normalize,
        )
        self.use_running_average = use_running_average
        self._window_count = 0
        self._avg = torch.zeros(window_size, device=device)

    def __call__(self, state: SimState, _: Any = None) -> torch.Tensor:
        """Update HFACF with new state.

        Args:
            state: Current simulation state
            _: Unused model argument (required property calculator interface)

        Returns:
            Tensor containing average HFACF
        """
        self.corr_calc.update(state)

        if self.corr_calc.buffers["heat_flux"].count == self.corr_calc.window_size:
            correlations = self.corr_calc.get_auto_correlations()
            # dims: (ndims, 1)
            hfacf = torch.mean(correlations["heat_flux"], dim=(1, 2))

            self._window_count += 1

            if self.use_running_average:
                factor = 1.0 / self._window_count
                self._avg += (hfacf - self._avg) * factor
            else:
                self._avg = hfacf

            self.corr_calc.reset()

        return torch.tensor([self._window_count], device=state.device)

    @property
    def hfacf(self) -> torch.Tensor:
        """Current HFACF result."""
        return self._avg


class RadialDistributionFunction:
    """On-the-fly radial distribution function calculator.

    Accumulates pair-distance histograms over trajectory frames and
    returns the normalized g(r) averaged over all sampled frames.
    Handles periodic boundary conditions via the minimum-image convention
    using fractional coordinates.

    Supports optional species-pair filtering for multi-element systems.

    Using ``RadialDistributionFunction`` with
    :class:`~ts.trajectory.TrajectoryReporter`::

        rdf_calc = RadialDistributionFunction(
            r_max=8.0,
            n_bins=200,
            device=device,
        )
        reporter = TrajectoryReporter(
            "simulation.h5",
            prop_calculators={10: {"rdf": rdf_calc}},
        )
        # After simulation:
        r, gr = rdf_calc.gr

    """

    def __init__(
        self,
        *,
        r_max: float,
        n_bins: int,
        device: torch.device,
        species_pair: tuple[int, int] | None = None,
    ) -> None:
        """Initialize RDF calculator.

        Args:
            r_max: Maximum pair distance to histogram (Å).
            n_bins: Number of histogram bins.
            device: Computation device.
            species_pair: If given as ``(Zi, Zj)``, only count pairs between
                those atomic numbers. ``None`` counts all pairs.
        """
        self.r_max = r_max
        self.n_bins = n_bins
        self.device = device
        self.species_pair = species_pair

        edges = torch.linspace(0.0, r_max, n_bins + 1, device=device, dtype=torch.float64)
        self.r_edges = edges
        self.r_centers = 0.5 * (edges[:-1] + edges[1:])

        self._hist = torch.zeros(n_bins, device=device, dtype=torch.float64)
        self._frame_count = 0
        self._n_atoms_sum = 0
        self._volume_sum = 0.0

    def __call__(self, state: SimState, _: Any = None) -> torch.Tensor:
        """Accumulate one frame into the histogram.

        Args:
            state: Current simulation state (single system).
            _: Unused model argument.

        Returns:
            Tensor containing the current frame count.
        """
        pos = state.positions.to(torch.float64)   # (n, 3)
        cell = state.cell[0].to(torch.float64)    # (3, 3), rows = lattice vectors
        n = pos.shape[0]

        # All pairwise displacement vectors: (n, n, 3)
        dr = pos.unsqueeze(0) - pos.unsqueeze(1)

        # Minimum image via fractional coordinates: s = dr @ cell⁻¹
        cell_inv = torch.linalg.inv(cell)
        dr_frac = dr @ cell_inv.T
        dr_frac = dr_frac - torch.round(dr_frac)
        dr_cart = dr_frac @ cell.T

        dists = torch.linalg.norm(dr_cart, dim=-1)  # (n, n)

        # Upper-triangle mask (i < j) to avoid double-counting
        mask = torch.triu(torch.ones(n, n, device=self.device, dtype=torch.bool), diagonal=1)

        if self.species_pair is not None:
            z = state.atomic_numbers
            zi, zj = self.species_pair
            sp_mask = (
                (z.unsqueeze(0) == zi) & (z.unsqueeze(1) == zj)
            ) | (
                (z.unsqueeze(0) == zj) & (z.unsqueeze(1) == zi)
            )
            mask = mask & sp_mask

        dists_flat = dists[mask]
        dists_flat = dists_flat[dists_flat < self.r_max]

        hist = torch.histc(
            dists_flat.float(), bins=self.n_bins, min=0.0, max=float(self.r_max)
        )
        # Multiply by 2 so both (i,j) and (j,i) are counted
        self._hist += hist.to(torch.float64) * 2

        self._frame_count += 1
        self._n_atoms_sum += n
        self._volume_sum += torch.det(cell).abs().item()

        return torch.tensor([self._frame_count], device=state.device)

    @property
    def gr(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized g(r) averaged over accumulated frames.

        Returns:
            Tuple ``(r_centers, g_values)`` on CPU, both shape ``(n_bins,)``.
            ``r_centers`` is in Å.  Returns zeros if no frames accumulated.
        """
        if self._frame_count == 0:
            return self.r_centers.cpu(), torch.zeros(self.n_bins)

        avg_n = self._n_atoms_sum / self._frame_count
        avg_vol = self._volume_sum / self._frame_count
        rho = avg_n / avg_vol  # Å⁻³

        shell_vol = (4.0 / 3.0) * torch.pi * (
            self.r_edges[1:] ** 3 - self.r_edges[:-1] ** 3
        )
        ideal = avg_n * rho * shell_vol

        g = (self._hist / self._frame_count) / ideal
        return self.r_centers.cpu(), g.cpu()

    def reset(self) -> None:
        """Reset all accumulated data."""
        self._hist.zero_()
        self._frame_count = 0
        self._n_atoms_sum = 0
        self._volume_sum = 0.0


class PressureAutoCorrelation:
    """Calculator for pressure tensor autocorrelation function (PACF) for viscosity.

    Computes the Green-Kubo shear viscosity via:

        η = (V / kT) ∫₀^∞ < P_αβ(0) · P_αβ(t) > dt

    where the average runs over the three independent off-diagonal components
    Pxy, Pxz, Pyz.  Uses an FFT-based circular-window estimator identical in
    spirit to ``VelocityAutoCorrelation``.

    Requires the model to return ``'stress'`` [n_systems, 3, 3] in eV/Å³.

    Using ``PressureAutoCorrelation`` with
    :class:`~ts.trajectory.TrajectoryReporter`::

        pacf_calc = PressureAutoCorrelation(
            model=model,
            window_size=500,
            temperature=600.0,
            device=device,
        )
        reporter = TrajectoryReporter(
            "simulation.h5",
            prop_calculators={1: {"pacf": pacf_calc}},
        )
        # After simulation:
        eta = pacf_calc.viscosity(timestep_ps=0.001)

    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        window_size: int,
        temperature: float,
        device: torch.device,
        use_running_average: bool = True,
    ) -> None:
        """Initialize PACF calculator.

        Args:
            model: MLIP model; must return ``'stress'`` tensor.
            window_size: Correlation window length (steps).
            temperature: Simulation temperature in Kelvin (for viscosity prefactor).
            device: Computation device.
            use_running_average: Accumulate a running average across windows.
        """
        self.model = model
        self.temperature = temperature
        self.window_size = window_size
        self.device = device
        self.use_running_average = use_running_average

        # Circular buffer storing (Pxy, Pxz, Pyz) at each step: shape (window_size, 3)
        self._buffer = torch.zeros(window_size, 3, device=device, dtype=torch.float64)
        self._pos = 0
        self._count = 0

        self._window_count = 0
        self._avg = torch.zeros(window_size, device=device, dtype=torch.float64)
        self._volume_sum = 0.0
        self._call_count = 0

    def _shear_components(self, state: SimState) -> torch.Tensor:
        """Return (Pxy, Pxz, Pyz) in eV/Å³ for system 0."""
        out = self.model(state)
        P = compute_instantaneous_pressure_tensor(
            momenta=state.momenta,
            masses=state.masses,
            system_idx=state.system_idx,
            stress=out["stress"],
            volumes=state.volume,
        )  # (n_systems, 3, 3)
        return torch.stack([P[0, 0, 1], P[0, 0, 2], P[0, 1, 2]]).to(torch.float64)

    def __call__(self, state: SimState, _: Any = None) -> torch.Tensor:
        """Record one pressure sample and update the running PACF.

        Args:
            state: Current simulation state.
            _: Unused model argument.

        Returns:
            Tensor containing the current completed window count.
        """
        shear = self._shear_components(state).detach()

        self._buffer[self._pos] = shear
        self._pos = (self._pos + 1) % self.window_size
        self._count = min(self._count + 1, self.window_size)
        self._volume_sum += state.volume[0].item()
        self._call_count += 1

        if self._count == self.window_size:
            # Re-order buffer into chronological order
            if self._pos == 0:
                series = self._buffer.clone()
            else:
                series = torch.cat([self._buffer[self._pos:], self._buffer[: self._pos]])

            # FFT autocorrelation averaged over the 3 shear components
            n = self.window_size
            pacf = torch.zeros(n, device=self.device, dtype=torch.float64)
            for k in range(3):
                x = series[:, k]
                f = torch.fft.rfft(x, n=2 * n)
                acf_k = torch.fft.irfft(f * f.conj(), n=2 * n)[:n].real / n
                pacf += acf_k
            pacf /= 3.0

            self._window_count += 1
            if self.use_running_average:
                self._avg += (pacf - self._avg) / self._window_count
            else:
                self._avg = pacf

            # Reset for next window
            self._buffer.zero_()
            self._pos = 0
            self._count = 0

        return torch.tensor([self._window_count], device=state.device)

    @property
    def pacf(self) -> torch.Tensor:
        """Current PACF in (eV/Å³)²."""
        return self._avg

    def viscosity(self, timestep_ps: float) -> float:
        """Integrate the PACF to give shear viscosity.

        Uses the Green-Kubo relation with trapezoidal integration:

            η = (V / kT) ∫ < P_αβ(0) P_αβ(t) > dt

        Args:
            timestep_ps: MD timestep in picoseconds.

        Returns:
            Shear viscosity in Pa·s.
        """
        avg_vol_ang3 = self._volume_sum / max(self._call_count, 1)
        V_m3 = avg_vol_ang3 * 1e-30
        kT_J = 1.380649e-23 * self.temperature
        dt_s = timestep_ps * 1e-12
        # 1 eV/Å³ = 1.60218e-19 J / 1e-30 m³ = 1.60218e11 Pa
        eV_ang3_to_Pa = 1.60218e-19 / 1e-30
        pacf_Pa2 = self.pacf.cpu().numpy() * eV_ang3_to_Pa**2
        return float((V_m3 / kT_J) * np.trapz(pacf_Pa2, dx=dt_s))
