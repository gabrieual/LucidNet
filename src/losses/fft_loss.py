"""2D Fourier Amplitude & Phase Frequency Loss (float32 shielded)."""

import torch
import torch.nn as nn


class ComplexFFTLoss(nn.Module):
    """
    Computes L1 distance between real and imaginary components of the 2D Real FFT
    in forced float32 precision to recover high-frequency MTF details degraded by
    atmospheric turbulence without AMP overflow.
    """

    def __init__(self) -> None:
        super().__init__()

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim == 5:
            b, t, c, h, w = pred.shape
            pred = pred.reshape(b * t, c, h, w)
            target = target.reshape(b * t, c, h, w)

        pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
        target_fft = torch.fft.rfft2(target.float(), norm="ortho")

        loss_real = torch.abs(pred_fft.real - target_fft.real).mean()
        loss_imag = torch.abs(pred_fft.imag - target_fft.imag).mean()
        return loss_real + loss_imag
