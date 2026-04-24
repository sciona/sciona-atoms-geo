"""Ghost witnesses for GSD-aware augmentation atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_gsd_aware_random_crop(
    image: AbstractArray,
    gsd: float,
    target_gsd: float,
    crop_size: int,
) -> AbstractArray:
    """A GSD-aware crop returns an image-like array."""
    return AbstractArray(shape=("crop_size", "crop_size"), dtype=image.dtype)


def witness_gsd_aware_shift_scale_rotate(
    image: AbstractArray,
    gsd: float,
    shift_limit: float,
    scale_limit: float,
    rotate_limit: float,
    rng: object,
) -> tuple[AbstractArray, float]:
    """Affine transform preserves image shape and returns an updated GSD."""
    return AbstractArray(shape=image.shape, dtype=image.dtype), gsd
