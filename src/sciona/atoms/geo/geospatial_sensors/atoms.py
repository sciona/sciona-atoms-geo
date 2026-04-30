"""Geospatial, GNSS, and inertial sensor fusion atoms."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_apply_asymmetric_bias_correction,
    witness_correct_clock_bias,
    witness_detect_steps,
    witness_ecef_to_enu,
    witness_ecef_to_lla,
    witness_estimate_step_length_weinberg,
    witness_filter_by_cn0,
    witness_filter_multipath,
    witness_integrate_heading,
    witness_lla_to_ecef,
    witness_pdr_position_update,
    witness_resample_to_gsd,
    witness_rts_smooth,
    witness_snap_to_nearest,
)


_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)
_WGS84_EP2 = (_WGS84_A**2 - _WGS84_B**2) / (_WGS84_B**2)
_SPEED_OF_LIGHT = 299792458.0


def _finite_array(values: NDArray[np.float64]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(array.size >= 1 and np.all(np.isfinite(array)))


def _same_shape(*arrays: NDArray[np.float64]) -> bool:
    shapes = [np.asarray(array).shape for array in arrays]
    return bool(len(shapes) >= 1 and all(shape == shapes[0] for shape in shapes))


def _lat_lon_valid(lat: NDArray[np.float64], lon: NDArray[np.float64]) -> bool:
    lat_array = np.asarray(lat, dtype=np.float64)
    lon_array = np.asarray(lon, dtype=np.float64)
    return bool(
        _same_shape(lat_array, lon_array)
        and np.all(np.isfinite(lat_array))
        and np.all(np.isfinite(lon_array))
        and np.all(lat_array >= -90.0)
        and np.all(lat_array <= 90.0)
        and np.all(lon_array >= -180.0)
        and np.all(lon_array <= 180.0)
    )


def _finite_same_shape(*arrays: NDArray[np.float64]) -> bool:
    return bool(_same_shape(*arrays) and all(_finite_array(np.asarray(array, dtype=np.float64)) for array in arrays))


def _matrix_sequence_valid(values: NDArray[np.float64], ndim: int) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(array.ndim == ndim and array.shape[0] >= 1 and np.all(np.isfinite(array)))


@register_atom(witness_lla_to_ecef)
@icontract.require(lambda lat, lon: _lat_lon_valid(lat, lon), "lat/lon must be finite WGS84 degree arrays")
@icontract.require(lambda lat, lon, alt: _same_shape(lat, lon, alt), "lat, lon, and alt must share shape")
@icontract.require(lambda alt: _finite_array(alt), "alt must be finite")
@icontract.ensure(lambda result, lat: all(part.shape == np.asarray(lat).shape for part in result), "ECEF outputs must preserve shape")
@icontract.ensure(lambda result: all(np.all(np.isfinite(part)) for part in result), "ECEF outputs must be finite")
def lla_to_ecef(
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    alt: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert WGS84 latitude, longitude, and altitude to ECEF coordinates."""
    lat_rad = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lon_rad = np.deg2rad(np.asarray(lon, dtype=np.float64))
    alt_m = np.asarray(alt, dtype=np.float64)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    prime_vertical = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (prime_vertical + alt_m) * cos_lat * np.cos(lon_rad)
    y = (prime_vertical + alt_m) * cos_lat * np.sin(lon_rad)
    z = (prime_vertical * (1.0 - _WGS84_E2) + alt_m) * sin_lat
    return x.astype(np.float64), y.astype(np.float64), z.astype(np.float64)


@register_atom(witness_ecef_to_lla)
@icontract.require(lambda x, y, z: _finite_same_shape(x, y, z), "ECEF coordinates must be finite aligned arrays")
@icontract.ensure(lambda result, x: all(part.shape == np.asarray(x).shape for part in result), "LLA outputs must preserve shape")
@icontract.ensure(lambda result: np.all(result[0] >= -90.0) and np.all(result[0] <= 90.0), "latitude must be bounded")
@icontract.ensure(lambda result: np.all(result[1] >= -180.0) and np.all(result[1] <= 180.0), "longitude must be bounded")
def ecef_to_lla(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert ECEF coordinates to WGS84 latitude, longitude, and altitude."""
    x_m = np.asarray(x, dtype=np.float64)
    y_m = np.asarray(y, dtype=np.float64)
    z_m = np.asarray(z, dtype=np.float64)
    p = np.sqrt(x_m * x_m + y_m * y_m)
    theta = np.arctan2(z_m * _WGS84_A, p * _WGS84_B)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    lat_rad = np.arctan2(
        z_m + _WGS84_EP2 * _WGS84_B * sin_theta**3,
        p - _WGS84_E2 * _WGS84_A * cos_theta**3,
    )
    lon_rad = np.arctan2(y_m, x_m)
    sin_lat = np.sin(lat_rad)
    prime_vertical = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    alt = p / np.maximum(np.cos(lat_rad), 1e-15) - prime_vertical
    lat = np.rad2deg(lat_rad)
    lon = ((np.rad2deg(lon_rad) + 180.0) % 360.0) - 180.0
    return lat.astype(np.float64), lon.astype(np.float64), alt.astype(np.float64)


@register_atom(witness_ecef_to_enu)
@icontract.require(lambda x, y, z: _finite_same_shape(x, y, z), "ECEF coordinates must be finite aligned arrays")
@icontract.require(lambda ref_lat: -90.0 <= float(ref_lat) <= 90.0, "ref_lat must be a valid latitude")
@icontract.require(lambda ref_lon: -180.0 <= float(ref_lon) <= 180.0, "ref_lon must be a valid longitude")
@icontract.require(lambda ref_alt: np.isfinite(float(ref_alt)), "ref_alt must be finite")
@icontract.ensure(lambda result, x: all(part.shape == np.asarray(x).shape for part in result), "ENU outputs must preserve shape")
@icontract.ensure(lambda result: all(np.all(np.isfinite(part)) for part in result), "ENU outputs must be finite")
def ecef_to_enu(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    ref_lat: float,
    ref_lon: float,
    ref_alt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Project ECEF coordinates into a local east-north-up tangent plane."""
    ref_x, ref_y, ref_z = lla_to_ecef(
        np.array([float(ref_lat)], dtype=np.float64),
        np.array([float(ref_lon)], dtype=np.float64),
        np.array([float(ref_alt)], dtype=np.float64),
    )
    dx = np.asarray(x, dtype=np.float64) - float(ref_x[0])
    dy = np.asarray(y, dtype=np.float64) - float(ref_y[0])
    dz = np.asarray(z, dtype=np.float64) - float(ref_z[0])
    lat_rad = np.deg2rad(float(ref_lat))
    lon_rad = np.deg2rad(float(ref_lon))
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return east.astype(np.float64), north.astype(np.float64), up.astype(np.float64)


@register_atom(witness_correct_clock_bias)
@icontract.require(lambda pseudoranges, sat_clock_bias, rx_clock_bias: _finite_same_shape(pseudoranges, sat_clock_bias, rx_clock_bias), "GNSS clock arrays must align")
@icontract.ensure(lambda result, pseudoranges: result.shape == np.asarray(pseudoranges).shape, "corrected pseudoranges must preserve shape")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "corrected pseudoranges must be finite")
def correct_clock_bias(
    pseudoranges: NDArray[np.float64],
    sat_clock_bias: NDArray[np.float64],
    rx_clock_bias: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Correct pseudoranges for receiver and satellite clock biases in seconds."""
    return (
        np.asarray(pseudoranges, dtype=np.float64)
        - _SPEED_OF_LIGHT * (np.asarray(rx_clock_bias, dtype=np.float64) - np.asarray(sat_clock_bias, dtype=np.float64))
    ).astype(np.float64)


@register_atom(witness_filter_by_cn0)
@icontract.require(lambda measurements, cn0: _finite_same_shape(measurements, cn0), "measurements and cn0 must align")
@icontract.require(lambda threshold: np.isfinite(float(threshold)), "threshold must be finite")
@icontract.ensure(lambda result, measurements: result.shape == np.asarray(measurements).shape, "filtered measurements must preserve shape")
def filter_by_cn0(
    measurements: NDArray[np.float64],
    cn0: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.float64]:
    """Mask GNSS measurements whose carrier-to-noise density is below a threshold."""
    values = np.asarray(measurements, dtype=np.float64)
    return np.where(np.asarray(cn0, dtype=np.float64) >= float(threshold), values, np.nan).astype(np.float64)


@register_atom(witness_filter_multipath)
@icontract.require(lambda measurements, multipath_indicator: _finite_same_shape(measurements, multipath_indicator), "measurements and multipath indicators must align")
@icontract.require(lambda threshold: np.isfinite(float(threshold)) and float(threshold) >= 0.0, "threshold must be finite and non-negative")
@icontract.ensure(lambda result, measurements: result.shape == np.asarray(measurements).shape, "filtered measurements must preserve shape")
def filter_multipath(
    measurements: NDArray[np.float64],
    multipath_indicator: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.float64]:
    """Mask measurements whose multipath indicator exceeds a threshold."""
    values = np.asarray(measurements, dtype=np.float64)
    return np.where(np.abs(np.asarray(multipath_indicator, dtype=np.float64)) <= float(threshold), values, np.nan).astype(np.float64)


@register_atom(witness_detect_steps)
@icontract.require(lambda accel_magnitude: _finite_array(accel_magnitude) and np.asarray(accel_magnitude).ndim == 1, "accel_magnitude must be a finite 1-D array")
@icontract.require(lambda threshold: np.isfinite(float(threshold)), "threshold must be finite")
@icontract.require(lambda prominence: np.isfinite(float(prominence)) and float(prominence) >= 0.0, "prominence must be finite and non-negative")
@icontract.require(lambda min_distance: isinstance(min_distance, int) and min_distance >= 1, "min_distance must be positive")
@icontract.ensure(lambda result: result.ndim == 1 and np.all(result >= 0), "step indices must be a non-negative vector")
def detect_steps(
    accel_magnitude: NDArray[np.float64],
    threshold: float,
    prominence: float,
    min_distance: int,
) -> NDArray[np.int64]:
    """Detect step events as constrained peaks in acceleration magnitude."""
    peaks, _ = find_peaks(
        np.asarray(accel_magnitude, dtype=np.float64),
        height=float(threshold),
        prominence=float(prominence),
        distance=int(min_distance),
    )
    return peaks.astype(np.int64)


@register_atom(witness_estimate_step_length_weinberg)
@icontract.require(lambda accel_z: _finite_array(accel_z) and np.asarray(accel_z).ndim == 1, "accel_z must be a finite 1-D array")
@icontract.require(lambda accel_z, step_indices: np.asarray(step_indices).ndim == 1 and np.asarray(step_indices).size >= 2 and np.all(np.diff(np.asarray(step_indices)) > 0) and np.all(np.asarray(step_indices) >= 0) and np.all(np.asarray(step_indices) < np.asarray(accel_z).shape[0]), "step_indices must be sorted valid bounds")
@icontract.require(lambda k_constant: np.isfinite(float(k_constant)) and float(k_constant) > 0.0, "k_constant must be positive")
@icontract.ensure(lambda result, step_indices: result.shape == (np.asarray(step_indices).shape[0] - 1,), "one stride length per consecutive step interval")
@icontract.ensure(lambda result: np.all(np.isfinite(result)) and np.all(result >= 0.0), "step lengths must be finite and non-negative")
def estimate_step_length_weinberg(
    accel_z: NDArray[np.float64],
    step_indices: NDArray[np.int64],
    k_constant: float,
) -> NDArray[np.float64]:
    """Estimate stride lengths from vertical acceleration range per step cycle."""
    accel = np.asarray(accel_z, dtype=np.float64)
    steps = np.asarray(step_indices, dtype=np.int64)
    lengths = np.empty(steps.size - 1, dtype=np.float64)
    for idx in range(steps.size - 1):
        segment = accel[steps[idx] : steps[idx + 1] + 1]
        lengths[idx] = float(k_constant) * np.power(float(np.max(segment) - np.min(segment)), 0.25)
    return lengths


@register_atom(witness_integrate_heading)
@icontract.require(lambda gyro_z: _finite_array(gyro_z) and np.asarray(gyro_z).ndim == 1, "gyro_z must be a finite 1-D array")
@icontract.require(lambda dt: np.isfinite(float(dt)) and float(dt) > 0.0, "dt must be positive")
@icontract.require(lambda initial_heading: np.isfinite(float(initial_heading)), "initial_heading must be finite")
@icontract.ensure(lambda result, gyro_z: result.shape == np.asarray(gyro_z).shape, "heading output must preserve shape")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "heading output must be finite")
def integrate_heading(
    gyro_z: NDArray[np.float64],
    dt: float,
    initial_heading: float = 0.0,
) -> NDArray[np.float64]:
    """Integrate z-axis angular velocity into heading with trapezoidal steps."""
    rates = np.asarray(gyro_z, dtype=np.float64)
    headings = np.empty_like(rates, dtype=np.float64)
    headings[0] = float(initial_heading)
    if rates.size > 1:
        increments = 0.5 * (rates[1:] + rates[:-1]) * float(dt)
        headings[1:] = float(initial_heading) + np.cumsum(increments)
    return headings


@register_atom(witness_pdr_position_update)
@icontract.require(lambda step_lengths, headings: _finite_same_shape(step_lengths, headings), "step_lengths and headings must align")
@icontract.require(lambda step_lengths: np.asarray(step_lengths).ndim == 1 and np.all(np.asarray(step_lengths, dtype=np.float64) >= 0.0), "step_lengths must be a non-negative 1-D array")
@icontract.require(lambda initial_position: len(initial_position) == 2 and np.all(np.isfinite(np.asarray(initial_position, dtype=np.float64))), "initial_position must contain two finite coordinates")
@icontract.ensure(lambda result, step_lengths: result.shape == (np.asarray(step_lengths).shape[0] + 1, 2), "path must include initial point and one row per step")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "path coordinates must be finite")
def pdr_position_update(
    step_lengths: NDArray[np.float64],
    headings: NDArray[np.float64],
    initial_position: tuple[float, float],
) -> NDArray[np.float64]:
    """Accumulate a 2-D pedestrian dead-reckoning path from lengths and headings."""
    lengths = np.asarray(step_lengths, dtype=np.float64)
    angles = np.asarray(headings, dtype=np.float64)
    deltas = np.column_stack([lengths * np.sin(angles), lengths * np.cos(angles)])
    path = np.vstack([np.asarray(initial_position, dtype=np.float64), np.asarray(initial_position, dtype=np.float64) + np.cumsum(deltas, axis=0)])
    return path.astype(np.float64)


@register_atom(witness_snap_to_nearest)
@icontract.require(lambda trajectory: _matrix_sequence_valid(trajectory, 2), "trajectory must be a finite 2-D matrix")
@icontract.require(lambda grid_nodes: _matrix_sequence_valid(grid_nodes, 2), "grid_nodes must be a finite 2-D matrix")
@icontract.require(lambda trajectory, grid_nodes: np.asarray(trajectory).shape[1] == np.asarray(grid_nodes).shape[1], "trajectory and grid dimensions must match")
@icontract.ensure(lambda result, trajectory: result.shape == np.asarray(trajectory).shape, "snapped trajectory must preserve shape")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "snapped trajectory must be finite")
def snap_to_nearest(
    trajectory: NDArray[np.float64],
    grid_nodes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project trajectory points to their nearest allowed map/grid nodes."""
    tree = cKDTree(np.asarray(grid_nodes, dtype=np.float64))
    _, indices = tree.query(np.asarray(trajectory, dtype=np.float64))
    return np.asarray(grid_nodes, dtype=np.float64)[indices].astype(np.float64)


@register_atom(witness_resample_to_gsd)
@icontract.require(lambda image: np.asarray(image).ndim in {2, 3} and np.asarray(image).shape[0] >= 1 and np.asarray(image).shape[1] >= 1, "image must be a 2-D or 3-D spatial array")
@icontract.require(lambda current_gsd: np.isfinite(float(current_gsd)) and float(current_gsd) > 0.0, "current_gsd must be positive")
@icontract.require(lambda target_gsd: np.isfinite(float(target_gsd)) and float(target_gsd) > 0.0, "target_gsd must be positive")
@icontract.ensure(lambda result: result.ndim in {2, 3} and result.shape[0] >= 1 and result.shape[1] >= 1, "resampled image must remain spatial")
def resample_to_gsd(
    image: NDArray[np.float64],
    current_gsd: float,
    target_gsd: float,
) -> NDArray[np.float64]:
    """Resample a spatial image to a target ground sampling distance."""
    values = np.asarray(image)
    scale = float(current_gsd) / float(target_gsd)
    factors = (scale, scale) if values.ndim == 2 else (scale, scale, 1.0)
    return zoom(values, factors, order=1).astype(values.dtype, copy=False)


@register_atom(witness_rts_smooth)
@icontract.require(lambda filtered_states: _matrix_sequence_valid(filtered_states, 2), "filtered_states must be finite and 2-D")
@icontract.require(lambda filtered_covs, predicted_covs, transition_matrices: _matrix_sequence_valid(filtered_covs, 3) and _matrix_sequence_valid(predicted_covs, 3) and _matrix_sequence_valid(transition_matrices, 3), "covariance and transition histories must be finite 3-D arrays")
@icontract.require(lambda filtered_states, filtered_covs, predicted_states, predicted_covs, transition_matrices: np.asarray(filtered_states).shape == np.asarray(predicted_states).shape and np.asarray(filtered_covs).shape == np.asarray(predicted_covs).shape == np.asarray(transition_matrices).shape and np.asarray(filtered_covs).shape[0] == np.asarray(filtered_states).shape[0] and np.asarray(filtered_covs).shape[1] == np.asarray(filtered_states).shape[1] and np.asarray(filtered_covs).shape[2] == np.asarray(filtered_states).shape[1], "RTS histories must align")
@icontract.ensure(lambda result, filtered_states, filtered_covs: result[0].shape == np.asarray(filtered_states).shape and result[1].shape == np.asarray(filtered_covs).shape, "smoothed histories must preserve shapes")
@icontract.ensure(lambda result: np.all(np.isfinite(result[0])) and np.all(np.isfinite(result[1])), "smoothed histories must be finite")
def rts_smooth(
    filtered_states: NDArray[np.float64],
    filtered_covs: NDArray[np.float64],
    predicted_states: NDArray[np.float64],
    predicted_covs: NDArray[np.float64],
    transition_matrices: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run a fixed-interval Rauch-Tung-Striebel backward smoothing pass."""
    states = np.asarray(filtered_states, dtype=np.float64)
    covs = np.asarray(filtered_covs, dtype=np.float64)
    pred_states = np.asarray(predicted_states, dtype=np.float64)
    pred_covs = np.asarray(predicted_covs, dtype=np.float64)
    transitions = np.asarray(transition_matrices, dtype=np.float64)
    smoothed_states = states.copy()
    smoothed_covs = covs.copy()
    for idx in range(states.shape[0] - 2, -1, -1):
        gain = covs[idx] @ transitions[idx + 1].T @ np.linalg.pinv(pred_covs[idx + 1])
        smoothed_states[idx] = states[idx] + gain @ (smoothed_states[idx + 1] - pred_states[idx + 1])
        smoothed_covs[idx] = covs[idx] + gain @ (smoothed_covs[idx + 1] - pred_covs[idx + 1]) @ gain.T
        smoothed_covs[idx] = 0.5 * (smoothed_covs[idx] + smoothed_covs[idx].T)
    return smoothed_states, smoothed_covs


@register_atom(witness_apply_asymmetric_bias_correction)
@icontract.require(lambda predictions: _finite_array(predictions), "predictions must be finite")
@icontract.require(lambda condition_mask, predictions: np.asarray(condition_mask).shape == np.asarray(predictions).shape, "condition_mask must align with predictions")
@icontract.require(lambda multiplier: np.isfinite(float(multiplier)) and float(multiplier) > 0.0, "multiplier must be positive")
@icontract.ensure(lambda result, predictions: result.shape == np.asarray(predictions).shape, "corrected predictions must preserve shape")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "corrected predictions must be finite")
def apply_asymmetric_bias_correction(
    predictions: NDArray[np.float64],
    multiplier: float,
    condition_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Scale selected predictions by a positive asymmetric-loss correction factor."""
    values = np.asarray(predictions, dtype=np.float64)
    mask = np.asarray(condition_mask, dtype=np.bool_)
    return np.where(mask, values * float(multiplier), values).astype(np.float64)
