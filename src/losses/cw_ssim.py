"""Complex Wavelet SSIM (CW-SSIM) Loss insensitive to sub-pixel micro-shifts."""

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_complex_wavelet_bank(
    num_orientations: int = 6,
    kernel_size: int = 7,
    sigma: float = 1.6,
    wavelength: float = 3.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Builds 6-orientation complex quadrature wavelet filters (Real & Imaginary)."""
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    gauss = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))

    real_list, imag_list = [], []
    for k in range(num_orientations):
        theta = k * math.pi / num_orientations
        x_theta = xx * math.cos(theta) + yy * math.sin(theta)
        phase = 2.0 * math.pi * x_theta / wavelength
        real_k = gauss * torch.cos(phase)
        imag_k = gauss * torch.sin(phase)
        real_k = real_k - real_k.mean()
        imag_k = imag_k - imag_k.mean()
        norm = torch.sqrt(real_k.square().sum() + imag_k.square().sum()).clamp(min=1e-6)
        real_list.append(real_k / norm)
        imag_list.append(imag_k / norm)

    real_bank = torch.stack(real_list, dim=0).unsqueeze(1)  # [6, 1, K, K]
    imag_bank = torch.stack(imag_list, dim=0).unsqueeze(1)  # [6, 1, K, K]
    return real_bank, imag_bank


class ComplexWaveletSSIMLoss(nn.Module):
    """
    Multi-Scale Complex Wavelet Structural Similarity (CW-SSIM) Loss (Sampat et al., TIP 2009;
    Anantrasirichai et al., CLEAR TIP 2013):
        CW-SSIM(c_x, c_y) = (2 * |sum(c_x * conj(c_y))| + K) / (sum(|c_x|^2) + sum(|c_y|^2) + K)
    Evaluated across 3 dyadic scales and 6 orientations, making structural supervision
    invariant to small residual sub-pixel translations while strongly penalizing phase distortion.
    """

    def __init__(
        self,
        num_scales: int = 3,
        num_orientations: int = 6,
        window_size: int = 7,
        k_const: float = 1e-4,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        self.window_size = window_size
        self.k_const = k_const

        real_bank, imag_bank = _build_complex_wavelet_bank(num_orientations=num_orientations)
        self.register_buffer("real_bank", real_bank)
        self.register_buffer("imag_bank", imag_bank)
        self.register_buffer(
            "luma_weights", torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        )

    def _cw_ssim_single_scale(
        self, x_luma: torch.Tensor, y_luma: torch.Tensor
    ) -> torch.Tensor:
        pad = self.real_bank.shape[-1] // 2
        x_pad = F.pad(x_luma, (pad, pad, pad, pad), mode="reflect")
        y_pad = F.pad(y_luma, (pad, pad, pad, pad), mode="reflect")

        cx_r = F.conv2d(x_pad, self.real_bank)
        cx_i = F.conv2d(x_pad, self.imag_bank)
        cy_r = F.conv2d(y_pad, self.real_bank)
        cy_i = F.conv2d(y_pad, self.imag_bank)

        # Complex inner product c_x * conj(c_y) = (cx_r*cy_r + cx_i*cy_i) + j(cx_i*cy_r - cx_r*cy_i)
        prod_r = cx_r * cy_r + cx_i * cy_i
        prod_i = cx_i * cy_r - cx_r * cy_i

        mag_x_sq = cx_r.square() + cx_i.square()
        mag_y_sq = cy_r.square() + cy_i.square()

        w = self.window_size
        p_w = w // 2
        sum_prod_r = F.avg_pool2d(prod_r, kernel_size=w, stride=1, padding=p_w)
        sum_prod_i = F.avg_pool2d(prod_i, kernel_size=w, stride=1, padding=p_w)
        sum_mag_x = F.avg_pool2d(mag_x_sq, kernel_size=w, stride=1, padding=p_w)
        sum_mag_y = F.avg_pool2d(mag_y_sq, kernel_size=w, stride=1, padding=p_w)

        cross_abs = torch.sqrt(sum_prod_r.square() + sum_prod_i.square() + 1e-12)
        cw_ssim_map = (2.0 * cross_abs + self.k_const) / (
            sum_mag_x + sum_mag_y + self.k_const
        )
        return cw_ssim_map.mean()

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def compute_similarity(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Returns mean CW-SSIM score in [0, 1] across scales."""
        if pred.ndim == 5:
            b, t, c, h, w = pred.shape
            pred = pred.reshape(b * t, c, h, w)
            target = target.reshape(b * t, c, h, w)

        x_luma = (pred.float() * self.luma_weights).sum(dim=1, keepdim=True)
        y_luma = (target.float() * self.luma_weights).sum(dim=1, keepdim=True)

        scores = []
        for s in range(self.num_scales):
            if min(x_luma.shape[-2], x_luma.shape[-1]) < 16:
                break
            scores.append(self._cw_ssim_single_scale(x_luma, y_luma))
            if s < self.num_scales - 1:
                x_luma = F.avg_pool2d(x_luma, kernel_size=2, stride=2)
                y_luma = F.avg_pool2d(y_luma, kernel_size=2, stride=2)

        return torch.stack(scores).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Returns CW-SSIM loss: 1 - CW-SSIM(pred, target)."""
        return 1.0 - self.compute_similarity(pred, target)
