"""Ghost witnesses for geospatial loss atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_r2_regression_loss(
    predictions: AbstractArray,
    targets: AbstractArray,
    target_mean: float | None = None,
) -> float:
    """R2 regression loss maps paired arrays to a non-negative scalar."""
    return 0.0


def witness_circular_direction_loss(
    pred_sin: AbstractArray,
    pred_cos: AbstractArray,
    target_angle: AbstractArray,
) -> float:
    """Circular direction loss maps direction predictions to a scalar angle error."""
    return 0.0
