"""Gated Reference Update Module (GRUM) from DATUM."""

import torch
import torch.nn as nn


class GatedReferenceUpdateModule(nn.Module):
    """
    Convolutional GRU that recurrently updates the tilt-free reference feature r_t
    by combining the previous hidden reference r_{t-1} with the deep integrated embedding e_t.
    Enables tracking moving objects without ghosting while accumulating static background clarity.
    """

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.conv_z = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.conv_u = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.conv_r = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)

    def forward(self, r_prev: torch.Tensor, e_curr: torch.Tensor) -> torch.Tensor:
        cat_inp = torch.cat([r_prev, e_curr], dim=1)
        reset_gate = torch.sigmoid(self.conv_z(cat_inp))
        update_gate = torch.sigmoid(self.conv_u(cat_inp))
        candidate = torch.tanh(self.conv_r(torch.cat([r_prev * reset_gate, e_curr], dim=1)))
        r_next = update_gate * r_prev + (1.0 - update_gate) * candidate
        return r_next
