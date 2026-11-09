"""Visualization utilities for saving side-by-side restoration grids and estimated PSF kernels."""

from pathlib import Path
from typing import Optional
import torch
from torchvision.utils import make_grid, save_image


@torch.no_grad()
def save_comparison_grid(
    degraded: torch.Tensor,
    rectified: torch.Tensor,
    restored: torch.Tensor,
    gt: Optional[torch.Tensor],
    save_path: str | Path,
    frame_idx: int = 0,
) -> None:
    """
    Saves a visual comparison grid of a selected frame across the batch:
    [Degraded I_t | Tilt-Rectified I_rect | Restored J_hat | Ground-Truth J (if available)].
    """
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    t = degraded.shape[1]
    idx = min(max(0, frame_idx), t - 1)

    panels = [
        degraded[:, idx].detach().cpu().clamp(0.0, 1.0),
        rectified[:, idx].detach().cpu().clamp(0.0, 1.0),
        restored[:, idx].detach().cpu().clamp(0.0, 1.0),
    ]
    if gt is not None:
        panels.append(gt[:, idx].detach().cpu().clamp(0.0, 1.0))

    # Concatenate panels along batch dimension so each row is a sample
    num_cols = len(panels)
    stacked = torch.stack(panels, dim=1). flatten(0, 1)  # [B * num_cols, C, H, W]
    grid = make_grid(stacked, nrow=num_cols, padding=4, pad_value=1.0)
    save_image(grid, str(out))


@torch.no_grad()
def save_psf_grid(
    psf_kernels: Optional[torch.Tensor],
    save_path: str | Path,
    max_channels: int = 16,
) -> None:
    """
    Saves a normalized visualization of the learned M=96 Wiener PSF kernels K_hat_m (15x15)
    from Block 3 to inspect physical PSF convergence during training.
    """
    if psf_kernels is None:
        return
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # psf_kernels: [N, M, K, K] -> take first item in batch, first `max_channels` basis kernels
    kernels = psf_kernels[0, :max_channels].detach().cpu().unsqueeze(1)  # [M', 1, K, K]
    k_max = kernels.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    kernels_norm = kernels / k_max
    grid = make_grid(kernels_norm, nrow=4, padding=2, pad_value=0.5)
    save_image(grid, str(out))
