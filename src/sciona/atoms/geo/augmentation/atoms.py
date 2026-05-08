"""GSD-aware augmentation atoms for overhead imagery.

Extracts the geospatially aware augmentation logic from the 2nd-place Overhead
Geopose 2021 solution. The original source integrates these operations into
Albumentations transforms; the atoms below expose the reusable geometry in a
lightweight numpy/OpenCV form.

Source: geopose-2021-winners/2nd Place/geopose/augmentations.py (MIT)
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import witness_gsd_aware_random_crop, witness_gsd_aware_shift_scale_rotate

def _resize_with_scale(image: NDArray[np.generic], scale: float) -> NDArray[np.generic]:
    import cv2
    height, width = image.shape[:2]
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_CUBIC if scale >= 1.0 else cv2.INTER_AREA
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)

def _pad_to_minimum_size(image: NDArray[np.generic], crop_size: int) -> NDArray[np.generic]:
    import cv2
    height, width = image.shape[:2]
    pad_y = max(0, crop_size - height)
    pad_x = max(0, crop_size - width)
    if pad_x == 0 and pad_y == 0:
        return image

    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left
    return cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101)

@register_atom(witness_gsd_aware_random_crop)
@icontract.require(lambda image: image.ndim in (2, 3), "image must be 2-D or 3-D")
@icontract.require(lambda gsd: gsd > 0.0, "gsd must be positive")
@icontract.require(lambda target_gsd: target_gsd > 0.0, "target_gsd must be positive")
@icontract.require(lambda crop_size: crop_size >= 1, "crop_size must be at least 1")
@icontract.ensure(
    lambda result, crop_size: result.shape[0] == crop_size and result.shape[1] == crop_size,
    "cropped image must have the requested spatial shape",
)
def gsd_aware_random_crop(
    image: NDArray[np.generic],
    gsd: float,
    target_gsd: float,
    crop_size: int,
) -> NDArray[np.generic]:
    """Rescale to a target GSD before drawing a random square crop.

    Geopose uses GSD metadata to keep crops spatially meaningful across cities
    and sensors. This atom applies the same core idea: convert the image to the
    requested meters-per-pixel scale, then crop in pixel space.
    """
    image = np.asarray(image)
    scale = gsd / target_gsd
    resized = _resize_with_scale(image, scale)
    resized = _pad_to_minimum_size(resized, crop_size)

    height, width = resized.shape[:2]
    top = 0 if height == crop_size else int(np.random.randint(0, height - crop_size + 1))
    left = 0 if width == crop_size else int(np.random.randint(0, width - crop_size + 1))
    return resized[top : top + crop_size, left : left + crop_size].copy()

@register_atom(witness_gsd_aware_shift_scale_rotate)
@icontract.require(lambda image: image.ndim in (2, 3), "image must be 2-D or 3-D")
@icontract.require(lambda gsd: gsd > 0.0, "gsd must be positive")
@icontract.require(lambda shift_limit: shift_limit >= 0.0, "shift_limit must be non-negative")
@icontract.require(lambda scale_limit: scale_limit >= 0.0, "scale_limit must be non-negative")
@icontract.require(lambda rotate_limit: rotate_limit >= 0.0, "rotate_limit must be non-negative")
@icontract.ensure(
    lambda result, image: result[0].shape == image.shape,
    "transformed image must preserve the original shape",
)
@icontract.ensure(
    lambda result: np.isfinite(result[1]) and result[1] > 0.0,
    "updated gsd must be a positive finite scalar",
)
def gsd_aware_shift_scale_rotate(
    image: NDArray[np.generic],
    gsd: float,
    shift_limit: float,
    scale_limit: float,
    rotate_limit: float,
    rng: Generator,
) -> tuple[NDArray[np.generic], float]:
    import cv2
    """Apply an affine transform and propagate the updated GSD.

    This matches the Geopose transform semantics where image scale changes alter
    the effective meters-per-pixel value. Translation and rotation preserve GSD;
    the sampled scale factor updates it multiplicatively.
    """
    image = np.asarray(image)
    height, width = image.shape[:2]

    dx = float(rng.uniform(-shift_limit, shift_limit))
    dy = float(rng.uniform(-shift_limit, shift_limit))
    sampled_scale = max(1.0 + float(rng.uniform(-scale_limit, scale_limit)), 1e-6)
    angle_degrees = float(rng.uniform(-rotate_limit, rotate_limit))

    center = (width * 0.5, height * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, sampled_scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height

    transformed = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return transformed, float(gsd * sampled_scale)
