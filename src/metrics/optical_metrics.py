"""Optical & Turbulence Restoration Evaluation Metrics (PSNR, SSIM, CW-SSIM, T-LPIPS)."""

from typing import Dict
import torch
import torch.nn.functional as F

from src.losses.cw_ssim import ComplexWaveletSSIMLoss

_CW_SSIM_EVALUATOR = None


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Computes average Peak Signal-to-Noise Ratio (dB) across frames in [0, 1].
    Args:
        pred: [B, T, C, H, W] or [N, C, H, W]
        target: same shape as pred
    """
    if pred.ndim == 5:
        b, t, c, h, w = pred.shape
        pred = pred.reshape(b * t, c, h, w)
        target = target.reshape(b * t, c, h, w)

    mse = torch.mean((pred.float() - target.float()) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return float(psnr.mean().item())


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """Computes standard spatial SSIM in [0, 1]."""
    if pred.ndim == 5:
        b, t, c, h, w = pred.shape
        pred = pred.reshape(b * t, c, h, w)
        target = target.reshape(b * t, c, h, w)

    pred = pred.float()
    target = target.float()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pad = window_size // 2

    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=pad)

    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=pad) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=pad) - mu_y_sq
    sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu_xy

    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )
    return float(ssim_map.mean().item())


def compute_cw_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Computes Complex Wavelet SSIM (CW-SSIM) in [0, 1], robust to sub-pixel micro-jitter."""
    global _CW_SSIM_EVALUATOR
    if _CW_SSIM_EVALUATOR is None or next(_CW_SSIM_EVALUATOR.buffers()).device != pred.device:
        _CW_SSIM_EVALUATOR = ComplexWaveletSSIMLoss().to(pred.device)
    with torch.no_grad():
        return float(_CW_SSIM_EVALUATOR.compute_similarity(pred, target).item())


class MetricTracker:
    """Running average accumulator for validation and test metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], n: int = 1) -> None:
        for k, v in metrics.items():
            self._sums[k] = self._sums.get(k, 0.0) + float(v) * n
            self._counts[k] = self._counts.get(k, 0) + n

    def result(self) -> Dict[str, float]:
        return {
            k: self._sums[k] / max(1, self._counts[k]) for k in self._sums
        }
