"""Geospatial loss function atoms in pure numpy.

Extracts reusable loss primitives from the 1st-place Overhead Geopose 2021
solution. These atoms capture the framework-agnostic math behind the height
regression and direction/orientation losses used in the training pipeline.

Source: geopose-2021-winners/1st Place/training/losses.py (MIT)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import witness_circular_direction_loss, witness_r2_regression_loss


@register_atom(witness_r2_regression_loss)
@icontract.require(
    lambda predictions, targets: predictions.shape == targets.shape,
    "predictions and targets must have the same shape",
)
@icontract.require(lambda predictions: predictions.size >= 1, "predictions must be non-empty")
@icontract.require(
    lambda target_mean: target_mean is None or np.isfinite(target_mean),
    "target_mean must be finite when provided",
)
@icontract.ensure(
    lambda result: np.isfinite(result) and result >= 0.0,
    "R2 regression loss must be a non-negative finite scalar",
)
def r2_regression_loss(
    predictions: NDArray[np.float64],
    targets: NDArray[np.float64],
    target_mean: float | None = None,
) -> float:
    """Compute the Geopose R2-style regression loss.

    The upstream implementation optimizes the competition metric directly using
    ``ss_res / ss_tot`` rather than mean squared error. This makes the loss
    scale-invariant with respect to target variance, which is useful when the
    target distribution differs strongly across locations or cities.

    Args:
        predictions: Predicted regression outputs.
        targets: Ground-truth regression targets.
        target_mean: Optional precomputed mean of the target distribution.
            When omitted, the mean is computed from ``targets``.

    Returns:
        The scalar loss ``(ss_res + eps) / (ss_tot + eps)``.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    if target_mean is None:
        target_mean = float(np.mean(targets))

    eps = 1e-8
    ss_tot = np.sum((targets - target_mean) ** 2)
    ss_res = np.sum((targets - predictions) ** 2)
    return float((ss_res + eps) / (ss_tot + eps))


@register_atom(witness_circular_direction_loss)
@icontract.require(
    lambda pred_sin, pred_cos: pred_sin.shape == pred_cos.shape,
    "pred_sin and pred_cos must have the same shape",
)
@icontract.require(
    lambda pred_sin, target_angle: pred_sin.shape == target_angle.shape,
    "predictions and target_angle must have the same shape",
)
@icontract.require(lambda pred_sin: pred_sin.size >= 1, "predictions must be non-empty")
@icontract.ensure(
    lambda result: np.isfinite(result) and 0.0 <= result <= np.pi,
    "circular direction loss must lie in [0, pi]",
)
def circular_direction_loss(
    pred_sin: NDArray[np.float64],
    pred_cos: NDArray[np.float64],
    target_angle: NDArray[np.float64],
) -> float:
    """Compute mean angular error for heading/orientation prediction.

    The upstream angle loss compares predicted and target direction vectors,
    derives the cosine similarity, and converts that similarity to an angular
    distance using ``atan2(sin, cos)``. This atom keeps the same circular-error
    behavior while exposing a vectorized numpy interface.

    Args:
        pred_sin: Predicted sine component of the direction vector.
        pred_cos: Predicted cosine component of the direction vector.
        target_angle: Ground-truth target angles in radians.

    Returns:
        Mean angular distance in radians.
    """
    pred_sin = np.asarray(pred_sin, dtype=np.float64)
    pred_cos = np.asarray(pred_cos, dtype=np.float64)
    target_angle = np.asarray(target_angle, dtype=np.float64)

    target_sin = np.sin(target_angle)
    target_cos = np.cos(target_angle)

    pred_norm = np.hypot(pred_sin, pred_cos)
    target_norm = np.hypot(target_sin, target_cos)
    dot = pred_sin * target_sin + pred_cos * target_cos
    denom = pred_norm * target_norm
    cos_sim = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0.0)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    sin_term = np.sqrt(np.maximum(1.0 - cos_sim ** 2, 0.0))
    angular_error = np.arctan2(sin_term, cos_sim)
    return float(np.mean(angular_error))
