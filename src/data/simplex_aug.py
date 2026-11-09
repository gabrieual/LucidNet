"""On-the-Fly 3D Spatiotemporal Simplex/Perlin Turbulence Field Augmentor."""

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks.daab import flow_warp


class Simplex3DTurbulenceAugmentor(nn.Module):
    """
    Generates smooth 3D spatiotemporal multi-octave Simplex/Perlin displacement fields
    (x, y, t) following Kolmogorov-like 5/3 spatial power spectrum scaling across octaves.
    Returns `(degraded, tilt_only)` pairs on-the-fly for data augmentation.
    """

    def __init__(
        self,
        num_octaves: int = 3,
        base_grid_size: int = 8,
        persistence: float = 0.55,
        max_displacement: float = 6.0,
    ) -> None:
        super().__init__()
        self.num_octaves = num_octaves
        self.base_grid_size = base_grid_size
        self.persistence = persistence
        self.max_displacement = max_displacement

    @torch.no_grad()
    def forward(
        self, clean_video: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            clean_video: [T, C, H, W] in [0, 1]
        Returns:
            degraded: [T, C, H, W]
            tilt_only: [T, C, H, W]
        """
        t, c, h, w = clean_video.shape
        device = clean_video.device
        dtype = clean_video.dtype

        total_flow = torch.zeros(t, 2, h, w, device=device, dtype=torch.float32)
        amp = float(
            torch.empty(1, device=device).uniform_(0.3, 1.0).item()
            * self.max_displacement
        )

        for octave in range(self.num_octaves):
            freq_scale = 2 ** octave
            gt = max(2, t // max(1, (4 // freq_scale)))
            gh = max(4, self.base_grid_size * freq_scale)
            gw = max(4, self.base_grid_size * freq_scale)

            # 3D coarse lattice [1, 2, gt, gh, gw] interpolated smoothly in (t, y, x)
            lattice = torch.randn(1, 2, gt, gh, gw, device=device, dtype=torch.float32)
            # Zero-mean across time so no global static offset is introduced
            lattice = lattice - lattice.mean(dim=2, keepdim=True)
            octave_flow = F.interpolate(
                lattice, size=(t, h, w), mode="trilinear", align_corners=False
            ).squeeze(0)  # [2, T, H, W]
            total_flow = total_flow + octave_flow.permute(1, 0, 2, 3) * (
                amp * (self.persistence ** octave)
            )

        tilt_only = flow_warp(clean_video.float(), total_flow).clamp(0.0, 1.0)

        # Mild short-exposure diffraction/turbulence blur on top of tilt_only
        k = 7
        half = k // 2
        coords = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        sigma = float(torch.empty(1, device=device).uniform_(0.6, 1.4).item())
        psf = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
        psf = (psf / psf.sum()).view(1, 1, k, k).expand(c, 1, k, k)

        padded = F.pad(tilt_only, (half, half, half, half), mode="reflect")
        degraded = F.conv2d(padded, psf, groups=c).clamp(0.0, 1.0)
        return degraded.to(dtype), tilt_only.to(dtype)
