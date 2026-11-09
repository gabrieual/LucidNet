"""Phase-to-Space (P2S) Inspired Atmospheric Turbulence Simulator (On-the-Fly)."""

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks.daab import flow_warp


class PhysicsTurbulenceSimulator(nn.Module):
    """
    Simulates anisoplanatic atmospheric turbulence following the forward physical model:
        I_t(x) = B_t( T_t( J_t(x) ) ) + n_t(x)
    Produces:
      - `degraded`: Full turbulent sequence [T, C, H, W] (Tilt + Spatially-Varying Blur + Noise)
      - `tilt_only`: Intermediate tilt-distorted sequence T_t(J_t) without high-order PSF blur,
        used directly to supervise Branch A (L_tilt) of the Twin Decoder.
    Parametrized by Fried turbulence strength ratio D / r_0 in [1.5, 6.0].
    """

    def __init__(
        self,
        d_over_r0_range: Tuple[float, float] = (1.5, 5.0),
        corr_scale: int = 16,
        temporal_ar_rho: float = 0.75,
        psf_kernel_size: int = 15,
        noise_std_max: float = 0.015,
    ) -> None:
        super().__init__()
        self.d_over_r0_range = d_over_r0_range
        self.corr_scale = corr_scale
        self.temporal_ar_rho = temporal_ar_rho
        self.psf_kernel_size = psf_kernel_size
        self.noise_std_max = noise_std_max

    @torch.no_grad()
    def forward(
        self, clean_video: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Args:
            clean_video: Clean sequence J [T, C, H, W] in [0, 1]
        Returns:
            degraded: [T, C, H, W]
            tilt_only: [T, C, H, W]
            d_r0: sampled D/r_0 scalar strength
        """
        t, c, h, w = clean_video.shape
        device = clean_video.device
        dtype = clean_video.dtype

        low, high = self.d_over_r0_range
        d_r0 = float(torch.empty(1, device=device).uniform_(low, high).item())

        # 1. Generate temporally correlated, spatially smooth Kolmogorov-like tilt fields T_t
        gh = max(4, h // self.corr_scale)
        gw = max(4, w // self.corr_scale)
        tilt_amp = 0.85 * (d_r0 ** (5.0 / 6.0))

        coarse_tilts = []
        prev = torch.randn(1, 2, gh, gw, device=device, dtype=dtype)
        rho = self.temporal_ar_rho
        step_scale = math.sqrt(max(1e-6, 1.0 - rho * rho))
        for _ in range(t):
            innovation = torch.randn(1, 2, gh, gw, device=device, dtype=dtype)
            prev = rho * prev + step_scale * innovation
            coarse_tilts.append(prev)

        coarse_tilt_seq = torch.cat(coarse_tilts, dim=0) * tilt_amp  # [T, 2, gh, gw]
        tilt_fields = F.interpolate(
            coarse_tilt_seq, size=(h, w), mode="bicubic", align_corners=False
        )

        # Apply geometric tilt distortion T_t(J_t) -> produces `tilt_only` target
        tilt_only = flow_warp(clean_video, tilt_fields)

        # 2. Apply high-order Zernike / Atmospheric short-exposure PSF blur B_t
        # Approximate spatially varying blur via convex combination of 2 anisotropic Gaussian PSFs
        k = self.psf_kernel_size
        half = k // 2
        coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        sigma_base = 0.45 * (d_r0 ** (3.0 / 5.0))
        sigma_x1 = max(0.4, sigma_base * 1.25)
        sigma_y1 = max(0.4, sigma_base * 0.80)
        psf1 = torch.exp(-(xx.square() / (2 * sigma_x1**2) + yy.square() / (2 * sigma_y1**2)))
        psf1 = (psf1 / psf1.sum()).view(1, 1, k, k).expand(c, 1, k, k)

        sigma_x2 = max(0.4, sigma_base * 0.80)
        sigma_y2 = max(0.4, sigma_base * 1.25)
        psf2 = torch.exp(-(xx.square() / (2 * sigma_x2**2) + yy.square() / (2 * sigma_y2**2)))
        psf2 = (psf2 / psf2.sum()).view(1, 1, k, k).expand(c, 1, k, k)

        pad = half
        padded = F.pad(tilt_only, (pad, pad, pad, pad), mode="reflect")
        blur1 = F.conv2d(padded, psf1, groups=c)
        blur2 = F.conv2d(padded, psf2, groups=c)

        # Spatially & temporally varying mixing map alpha_t(x) in [0, 1] (simulates lucky regions)
        alpha_coarse = torch.sigmoid(
            torch.randn(t, 1, gh, gw, device=device, dtype=dtype) * 1.5
        )
        alpha = F.interpolate(alpha_coarse, size=(h, w), mode="bilinear", align_corners=False)

        # Occasional "lucky" sharp patches mixed in
        lucky_mask = (alpha > 0.82).to(dtype) * 0.35
        blurred = alpha * blur1 + (1.0 - alpha) * blur2
        blurred = (1.0 - lucky_mask) * blurred + lucky_mask * tilt_only

        # 3. Sensor readout / shot noise
        noise_std = float(torch.empty(1, device=device).uniform_(0.002, self.noise_std_max).item())
        degraded = torch.clamp(blurred + torch.randn_like(blurred) * noise_std, 0.0, 1.0)
        tilt_only = torch.clamp(tilt_only, 0.0, 1.0)

        return degraded, tilt_only, d_r0
