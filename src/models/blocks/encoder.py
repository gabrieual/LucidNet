"""Block 1.1: Shared Shallow Multi-Scale Feature Extractor (Scales 1/1, 1/2, 1/4)."""

from typing import Tuple
import torch
import torch.nn as nn

from src.contracts.interfaces import BaseMultiScaleEncoder
from src.models.registry import ENCODER_REGISTRY
from src.models.blocks.nafblock import NAFBlock


@ENCODER_REGISTRY.register("shallow_multiscale", "shallow_3scale")
class ShallowMultiScaleEncoder(BaseMultiScaleEncoder):
    """
    Extracts 3 hierarchical levels of features per frame (§2.2.1 of `arquitetura.md`):
      - Level 1 (Scale 1/1): c1 = 64
      - Level 2 (Scale 1/2): c2 = 96
      - Level 3 (Scale 1/4): c3 = 128
    Following the 2025 Survey (Hill et al.) energy analysis, the encoder uses lightweight
    single-block stages per scale to avoid redundant early-layer compute on blurred inputs.
    """

    def __init__(
        self,
        in_channels: int = 3,
        c1: int = 64,
        c2: int = 96,
        c3: int = 128,
    ) -> None:
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            NAFBlock(c1, dw_expand=1, ffn_expand=1),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            NAFBlock(c2, dw_expand=1, ffn_expand=1),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            NAFBlock(c3, dw_expand=2, ffn_expand=2),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return f1, f2, f3
