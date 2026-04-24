"""PyTorch ports for geospatial loss atoms."""

from __future__ import annotations

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised in environments without torch
    HAS_TORCH = False


def r2_regression_loss_torch(
    predictions: "torch.Tensor",
    targets: "torch.Tensor",
    target_mean: float | "torch.Tensor" | None = None,
) -> "torch.Tensor":
    """Differentiable R2 regression loss for autograd-enabled training."""
    if not HAS_TORCH:
        raise ImportError("torch required for GPU port")

    if target_mean is None:
        target_mean_tensor = targets.mean()
    elif torch.is_tensor(target_mean):
        target_mean_tensor = target_mean.to(device=targets.device, dtype=targets.dtype)
    else:
        target_mean_tensor = torch.tensor(target_mean, device=targets.device, dtype=targets.dtype)

    eps = torch.tensor(1e-8, device=targets.device, dtype=targets.dtype)
    ss_tot = torch.sum((targets - target_mean_tensor) ** 2)
    ss_res = torch.sum((targets - predictions) ** 2)
    return (ss_res + eps) / (ss_tot + eps)
