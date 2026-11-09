"""Block 4: Twin Decoder (Tilt Rectifier Branch A with CAB + NAFNet Restoration Branch B)."""

from typing import Tuple
import torch
import torch.nn as nn

from src.contracts.interfaces import BaseTwinDecoder
from src.models.registry import DECODER_REGISTRY
from src.models.blocks.nafblock import ChannelAttentionBlock, NAFBlock
from src.models.blocks.daab import flow_warp
from src.utils.tensor_ops import rearrange


@DECODER_REGISTRY.register("twin_naf_decoder")
class TwinNAFDecoder(BaseTwinDecoder):
    """
    Implements Section 2.5 of `arquitetura.md`:
      - Branch A (Geometric Tilt Rectifier T_hat_t^{-1}):
        Receives deep fused feature F_{lucky, t} (scale 1/4, c3=128) and shallow encoder
        features f_t^2 (scale 1/2, c2=96) and f_t^1 (scale 1/1, c1=64).
        Uses 2 stages of `Upsample + ChannelAttentionBlock (CAB)` to predict the dense
        inverse tilt field T_hat_t^{-1} at scale 1/1 and its downsampled version at scale 1/2.
        Rectifies f_t^2, f_t^1, and the RGB frame I_t BEFORE skip-connection fusion.
      - Branch B (Residual NAFNet Restoration):
        * Scale 1/2: Concatenates [F_{deconv, t} (64 ch), f_tilde_t^2 (96 ch)], reduces to
          96 channels via Conv1x1, and passes through 4 NAFBlocks.
        * Scale 1/1: Upsamples to HxW via PixelShuffle, concatenates with f_tilde_t^1 (64 ch),
          reduces to 64 channels via Conv1x1, and passes through 6 NAFBlocks.
        * Strict Residual Output: J_hat_t = I_{rect, t} + Conv3x3(NAFBlocks_out).
    """

    def __init__(
        self,
        c1: int = 64,
        c2: int = 96,
        c3: int = 128,
        deconv_channels: int = 64,
        num_blocks_s2: int = 4,
        num_blocks_s1: int = 6,
        max_tilt_displacement: float = 16.0,
    ) -> None:
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.deconv_channels = deconv_channels
        self.max_tilt_displacement = max_tilt_displacement

        # --- Branch A: 2 Stages of Upsample + ChannelAttentionBlock (CAB) (§2.5.1) ---
        self.tilt_up_s2 = nn.Sequential(
            nn.Conv2d(c3, c2 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )
        self.tilt_cab_s2 = nn.Sequential(
            nn.Conv2d(c2 * 2, c2, kernel_size=1),
            ChannelAttentionBlock(c2),
            ChannelAttentionBlock(c2),
        )
        self.tilt_head_s2 = nn.Conv2d(c2, 2, kernel_size=3, padding=1)

        self.tilt_up_s1 = nn.Sequential(
            nn.Conv2d(c2, c1 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )
        self.tilt_cab_s1 = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, kernel_size=1),
            ChannelAttentionBlock(c1),
            ChannelAttentionBlock(c1),
        )
        self.tilt_head_s1 = nn.Conv2d(c1, 2, kernel_size=3, padding=1)

        # Initialize final tilt layers to zero for identity start
        nn.init.zeros_(self.tilt_head_s2.weight)
        nn.init.zeros_(self.tilt_head_s2.bias)
        nn.init.zeros_(self.tilt_head_s1.weight)
        nn.init.zeros_(self.tilt_head_s1.bias)

        # --- Branch B Stage 1 (Scale 1/2: 4 NAFBlocks on 96 channels) (§2.5.2) ---
        self.fuse_s2 = nn.Conv2d(deconv_channels + c2, c2, kernel_size=1)
        self.decoder_s2 = nn.Sequential(*[NAFBlock(c2) for _ in range(num_blocks_s2)])

        # Upsample 1/2 -> 1/1
        self.up_s2_to_s1 = nn.Sequential(
            nn.Conv2d(c2, c1 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )

        # --- Branch B Stage 2 (Scale 1/1: 6 NAFBlocks on 64 channels) (§2.5.2) ---
        self.fuse_s1 = nn.Conv2d(c1 * 2, c1, kernel_size=1)
        self.decoder_s1 = nn.Sequential(*[NAFBlock(c1) for _ in range(num_blocks_s1)])
        self.rgb_head = nn.Conv2d(c1, 3, kernel_size=3, padding=1)

    def forward(
        self,
        frames: torch.Tensor,
        f1_seq: torch.Tensor,
        f2_seq: torch.Tensor,
        f_lucky_seq: torch.Tensor,
        f_deconv_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _, _, _ = frames.shape

        frames_flat = rearrange(frames, "b t c h w -> (b t) c h w")
        f1_flat = rearrange(f1_seq, "b t c h w -> (b t) c h w")
        f2_flat = rearrange(f2_seq, "b t c h w -> (b t) c h w")
        f_lucky_flat = rearrange(f_lucky_seq, "b t c h w -> (b t) c h w")
        f_deconv_flat = rearrange(f_deconv_seq, "b t c h w -> (b t) c h w")

        # --- Branch A: 2-stage Upsample + CAB Inverse Tilt Predictor ---
        t_feat_s2 = self.tilt_cab_s2(
            torch.cat([self.tilt_up_s2(f_lucky_flat), f2_flat], dim=1)
        )
        tilt_s2 = torch.tanh(self.tilt_head_s2(t_feat_s2)) * (
            self.max_tilt_displacement * 0.5
        )

        t_feat_s1 = self.tilt_cab_s1(
            torch.cat([self.tilt_up_s1(t_feat_s2), f1_flat], dim=1)
        )
        tilt_s1 = (
            torch.tanh(self.tilt_head_s1(t_feat_s1)) * self.max_tilt_displacement
        )

        # Rectify shallow skip connections and RGB frame using inverse tilt field
        f2_rect = flow_warp(f2_flat, tilt_s2)
        f1_rect = flow_warp(f1_flat, tilt_s1)
        rectified_rgb_flat = flow_warp(frames_flat, tilt_s1)

        # --- Branch B: Multi-scale NAFNet restoration (4 blocks at 1/2 + 6 blocks at 1/1) ---
        x_s2 = self.fuse_s2(torch.cat([f_deconv_flat, f2_rect], dim=1))
        x_s2 = self.decoder_s2(x_s2)

        x_s1_up = self.up_s2_to_s1(x_s2)
        x_s1 = self.fuse_s1(torch.cat([x_s1_up, f1_rect], dim=1))
        x_s1 = self.decoder_s1(x_s1)

        residual_rgb = self.rgb_head(x_s1)
        restored_flat = torch.clamp(rectified_rgb_flat + residual_rgb, 0.0, 1.0)

        restored = rearrange(restored_flat, "(b t) c h w -> b t c h w", b=b, t=t)
        rectified_rgb = rearrange(
            rectified_rgb_flat, "(b t) c h w -> b t c h w", b=b, t=t
        )
        inv_tilt_field = rearrange(tilt_s1, "(b t) c h w -> b t c h w", b=b, t=t)
        return restored, rectified_rgb, inv_tilt_field


@DECODER_REGISTRY.register("single_naf_decoder")
class SingleStageNAFDecoder(BaseTwinDecoder):
    """
    Ablation counterpart for Block 4 (Variants A, B, C without Twin Tilt Branch):
    Directly fuses unrectified shallow skip connections f1 and f2 with f_deconv
    using the same NAFBlock backbone (4 blocks at 1/2 + 6 blocks at 1/1).
    """

    def __init__(
        self,
        c1: int = 64,
        c2: int = 96,
        c3: int = 128,
        deconv_channels: int = 64,
        num_blocks_s2: int = 4,
        num_blocks_s1: int = 6,
        max_tilt_displacement: float = 16.0,
    ) -> None:
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.deconv_channels = deconv_channels

        self.fuse_s2 = nn.Conv2d(deconv_channels + c2, c2, kernel_size=1)
        self.decoder_s2 = nn.Sequential(*[NAFBlock(c2) for _ in range(num_blocks_s2)])
        self.up_s2_to_s1 = nn.Sequential(
            nn.Conv2d(c2, c1 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )
        self.fuse_s1 = nn.Conv2d(c1 * 2, c1, kernel_size=1)
        self.decoder_s1 = nn.Sequential(*[NAFBlock(c1) for _ in range(num_blocks_s1)])
        self.rgb_head = nn.Conv2d(c1, 3, kernel_size=3, padding=1)

    def forward(
        self,
        frames: torch.Tensor,
        f1_seq: torch.Tensor,
        f2_seq: torch.Tensor,
        f_lucky_seq: torch.Tensor,
        f_deconv_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _, h, w = frames.shape

        frames_flat = rearrange(frames, "b t c h w -> (b t) c h w")
        f1_flat = rearrange(f1_seq, "b t c h w -> (b t) c h w")
        f2_flat = rearrange(f2_seq, "b t c h w -> (b t) c h w")
        f_deconv_flat = rearrange(f_deconv_seq, "b t c h w -> (b t) c h w")

        x_s2 = self.fuse_s2(torch.cat([f_deconv_flat, f2_flat], dim=1))
        x_s2 = self.decoder_s2(x_s2)

        x_s1_up = self.up_s2_to_s1(x_s2)
        x_s1 = self.fuse_s1(torch.cat([x_s1_up, f1_flat], dim=1))
        x_s1 = self.decoder_s1(x_s1)

        residual_rgb = self.rgb_head(x_s1)
        restored_flat = torch.clamp(frames_flat + residual_rgb, 0.0, 1.0)

        restored = rearrange(restored_flat, "(b t) c h w -> b t c h w", b=b, t=t)
        zero_tilt = torch.zeros(
            b, t, 2, h, w, device=frames.device, dtype=frames.dtype
        )
        return restored, frames, zero_tilt
