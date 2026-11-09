"""Losses package for CWD-Net."""

from src.losses.charbonnier import CharbonnierLoss, TotalVariationLoss
from src.losses.fft_loss import ComplexFFTLoss
from src.losses.cw_ssim import ComplexWaveletSSIMLoss
from src.losses.composite_loss import CWDCompositeLoss

__all__ = [
    "CharbonnierLoss",
    "TotalVariationLoss",
    "ComplexFFTLoss",
    "ComplexWaveletSSIMLoss",
    "CWDCompositeLoss",
]
