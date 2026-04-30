"""Ghost witnesses for geospatial sensor atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_lla_to_ecef(lat: AbstractArray, lon: AbstractArray, alt: AbstractArray) -> tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Describe shape-preserving LLA to ECEF conversion."""
    return AbstractArray(shape=lat.shape, dtype="float64"), AbstractArray(shape=lat.shape, dtype="float64"), AbstractArray(shape=lat.shape, dtype="float64")


def witness_ecef_to_lla(x: AbstractArray, y: AbstractArray, z: AbstractArray) -> tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Describe shape-preserving ECEF to LLA conversion."""
    return AbstractArray(shape=x.shape, dtype="float64"), AbstractArray(shape=x.shape, dtype="float64"), AbstractArray(shape=x.shape, dtype="float64")


def witness_ecef_to_enu(
    x: AbstractArray,
    y: AbstractArray,
    z: AbstractArray,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float,
) -> tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Describe shape-preserving ECEF to ENU projection."""
    return AbstractArray(shape=x.shape, dtype="float64"), AbstractArray(shape=x.shape, dtype="float64"), AbstractArray(shape=x.shape, dtype="float64")


def witness_correct_clock_bias(
    pseudoranges: AbstractArray,
    sat_clock_bias: AbstractArray,
    rx_clock_bias: AbstractArray,
) -> AbstractArray:
    """Describe shape-preserving GNSS clock correction."""
    return AbstractArray(shape=pseudoranges.shape, dtype="float64")


def witness_filter_by_cn0(measurements: AbstractArray, cn0: AbstractArray, threshold: float) -> AbstractArray:
    """Describe shape-preserving C/N0 filtering."""
    return AbstractArray(shape=measurements.shape, dtype="float64")


def witness_filter_multipath(
    measurements: AbstractArray,
    multipath_indicator: AbstractArray,
    threshold: float,
) -> AbstractArray:
    """Describe shape-preserving multipath filtering."""
    return AbstractArray(shape=measurements.shape, dtype="float64")


def witness_detect_steps(
    accel_magnitude: AbstractArray,
    threshold: float,
    prominence: float,
    min_distance: int,
) -> AbstractArray:
    """Describe a vector of detected step indices."""
    return AbstractArray(shape=(accel_magnitude.shape[0],), dtype="int64")


def witness_estimate_step_length_weinberg(
    accel_z: AbstractArray,
    step_indices: AbstractArray,
    k_constant: float,
) -> AbstractArray:
    """Describe one stride length per adjacent step pair."""
    return AbstractArray(shape=(step_indices.shape[0],), dtype="float64")


def witness_integrate_heading(
    gyro_z: AbstractArray,
    dt: float,
    initial_heading: float = 0.0,
) -> AbstractArray:
    """Describe shape-preserving heading integration."""
    return AbstractArray(shape=gyro_z.shape, dtype="float64")


def witness_pdr_position_update(
    step_lengths: AbstractArray,
    headings: AbstractArray,
    initial_position: tuple[float, float],
) -> AbstractArray:
    """Describe a two-column path generated from step updates."""
    return AbstractArray(shape=(step_lengths.shape[0] + 1, 2), dtype="float64")


def witness_snap_to_nearest(trajectory: AbstractArray, grid_nodes: AbstractArray) -> AbstractArray:
    """Describe a snapped trajectory preserving input shape."""
    return AbstractArray(shape=trajectory.shape, dtype="float64")


def witness_resample_to_gsd(image: AbstractArray, current_gsd: float, target_gsd: float) -> AbstractArray:
    """Describe an image-like resampled array."""
    return AbstractArray(dtype=image.dtype)


def witness_rts_smooth(
    filtered_states: AbstractArray,
    filtered_covs: AbstractArray,
    predicted_states: AbstractArray,
    predicted_covs: AbstractArray,
    transition_matrices: AbstractArray,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe RTS smoothing outputs preserving state and covariance histories."""
    return AbstractArray(shape=filtered_states.shape, dtype="float64"), AbstractArray(shape=filtered_covs.shape, dtype="float64")


def witness_apply_asymmetric_bias_correction(
    predictions: AbstractArray,
    multiplier: float,
    condition_mask: AbstractArray,
) -> AbstractArray:
    """Describe shape-preserving prediction bias correction."""
    return AbstractArray(shape=predictions.shape, dtype="float64")
