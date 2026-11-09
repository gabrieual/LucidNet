"""Charbonnier L1 and Anisotropic Total Variation Losses."""

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """
    Differentiable Charbonnier L1 loss:
        L_char(x, y) = mean( sqrt( ||x - y||^2 + eps^2 ) )
    """

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps_sq = eps * eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred.float() - target.float()
        return torch.mean(torch.sqrt(diff * diff + self.eps_sq))


class TotalVariationLoss(nn.Module):
    """
    Anisotropic Spatial Total Variation (TV) regularization to suppress
    any residual deconvolution ringing in homogeneous sky/background regions:
        L_TV(x) = mean(|dx|) + mean(|dy|)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            x = x.reshape(b * t, c, h, w)
        dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return dh + dw
