"""Residual Dense Blocks (RDB) and Deep Integration Module (DIM) from DATUM."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (Zhang et al., CVPR 2018 / DATUM CVPR 2024).
    Also highlighted in the 2025 Survey (Vint et al.) as the top-performing CNN block for AT.
    """

    def __init__(self, channels: int = 128, growth_rate: int = 32, num_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        in_ch = channels
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, growth_rate, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                )
            )
            in_ch += growth_rate
        self.lff = nn.Conv2d(in_ch, channels, kernel_size=1)
        self.res_scale = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = [x]
        for layer in self.layers:
            out = layer(torch.cat(inputs, dim=1))
            inputs.append(out)
        fused = self.lff(torch.cat(inputs, dim=1))
        return x + fused * self.res_scale


class DeepIntegrationModule(nn.Module):
    """
    Deep Integration Module (DIM) from DATUM:
    Fuses registered feature f_t, hidden reference r_{t-1}, and raw feature f_t^3
    using a stack of RDBs to produce a tilt-free forward/backward embedding e_t.
    """

    def __init__(
        self,
        in_channels: int = 384,
        embed_channels: int = 128,
        num_rdbs: int = 4,
        growth_rate: int = 32,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_channels, kernel_size=1)
        self.rdbs = nn.ModuleList(
            [ResidualDenseBlock(embed_channels, growth_rate=growth_rate) for _ in range(num_rdbs)]
        )
        self.out_conv = nn.Conv2d(embed_channels, embed_channels, kernel_size=3, padding=1)
        self.use_checkpoint = use_checkpoint

    def forward(
        self, f_reg: torch.Tensor, r_prev: torch.Tensor, f_raw: torch.Tensor
    ) -> torch.Tensor:
        x = self.proj(torch.cat([f_reg, r_prev, f_raw], dim=1))
        res = x
        for rdb in self.rdbs:
            if self.use_checkpoint and x.requires_grad:
                x = checkpoint(rdb, x, use_reentrant=False)
            else:
                x = rdb(x)
        return res + self.out_conv(x)
