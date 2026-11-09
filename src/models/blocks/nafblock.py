"""Nonlinear Activation Free Network (NAFNet) and Channel Attention Blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-first 2D LayerNorm for [B, C, H, W] tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).square().mean(dim=1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class SimpleGate(nn.Module):
    """Splits channels in half and multiplies them element-wise (Chen et al., ECCV 2022)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    Nonlinear Activation Free Block (NAFNet, ECCV 2022) as proposed in proposta.pdf:
      - Depthwise convolution + SimpleGate
      - Simplified Channel Attention (SCA)
      - Pointwise Feed-Forward Network with SimpleGate
      - Learnable residual scaling parameters beta and gamma
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ) -> None:
        super().__init__()
        dw_ch = channels * dw_expand
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_ch, kernel_size=1)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, kernel_size=3, padding=1, groups=dw_ch)
        self.sg1 = SimpleGate()

        # Simplified Channel Attention (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, kernel_size=1),
        )
        self.conv3 = nn.Conv2d(dw_ch // 2, channels, kernel_size=1)

        # Feed-forward branch
        ffn_ch = channels * ffn_expand
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_ch, kernel_size=1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_ch // 2, channels, kernel_size=1)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg2(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class ChannelAttentionBlock(nn.Module):
    """Channel Attention Block (CAB) used in DATUM's Twin Decoder."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        res = res * self.ca(res)
        return x + res
