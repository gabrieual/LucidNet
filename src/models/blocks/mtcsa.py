"""Block 2: Multi-head Temporal-Channel Self-Attention (MTCSA) in 8x8 Local Windows."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.contracts.interfaces import BaseTemporalFusionBlock
from src.models.registry import FUSION_REGISTRY
from src.models.blocks.nafblock import LayerNorm2d
from src.utils.tensor_ops import rearrange


@FUSION_REGISTRY.register("mtcsa", "mtcsa_spectral_guided")
class MultiHeadTemporalChannelAttention(BaseTemporalFusionBlock):
    """
    Implements Section 2.3 of `arquitetura.md` (simulating CLEAR's Lucky Region Fusion
    and DATUM's MTCSA):
      1. Extracts a sliding temporal window of W=5 neighboring frames around each frame t
         (placing the focal frame t at the last slot for positional symmetry at boundaries).
      2. Modulates the W=5 embeddings by the classical sharpness priority P_tau:
             e_tilde_tau = e_tau * (1 + alpha_prior * Softmax_{tau in W}(tau_temp * P_tau))
      3. Stacks the 5 modulated frames into X in R^{B x (5 * C) x H/4 x W/4}, reduces channels
         with Conv1x1, applies DepthwiseSeparableConv3x3 to generate Q, K, V, and computes
         Multi-Head Temporal-Channel Self-Attention inside 8x8 local spatial windows!
    """

    def __init__(
        self,
        channels: int = 128,
        window_size: int = 5,
        spatial_window: int = 8,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.window_size = window_size
        self.spatial_window = spatial_window
        self.num_heads = num_heads

        # Learnable temperature and gain for the classical Lucky Imaging prior P_t
        self.prior_temp = nn.Parameter(torch.tensor(5.0))
        self.prior_alpha = nn.Parameter(torch.tensor(0.5))

        in_win_ch = channels * window_size
        self.norm_in = LayerNorm2d(in_win_ch)

        # Q projects from the focal-centric reduced representation, while K and V
        # preserve temporal-channel richness across the W=5 frames via grouped projections
        self.reduce_conv = nn.Conv2d(in_win_ch, channels, kernel_size=1)
        self.qkv_conv = nn.Conv2d(in_win_ch, channels * 3, kernel_size=1)
        self.qkv_dw = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
        )
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.ffn = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
        )

    def _gather_temporal_window(
        self, seq: torch.Tensor, focal_idx: int
    ) -> torch.Tensor:
        """
        Gather W neighboring frames around focal_idx, placing focal_idx at the last position
        (hard-coded focal-last positional ordering from DATUM Sec. 3.2.2 & §2.3.1).
        """
        t_total = seq.shape[1]
        half_w = self.window_size // 2
        indices = []
        for offset in range(-half_w, half_w + 1):
            if offset == 0:
                continue
            clamped = min(max(focal_idx + offset, 0), t_total - 1)
            indices.append(clamped)
        indices.append(focal_idx)  # Focal frame always last
        return seq[:, indices]  # [B, W, C, H, W]

    def _windowed_temporal_channel_attention(self, x_win_cat: torch.Tensor) -> torch.Tensor:
        """
        Computes temporal-channel self-attention within 8x8 spatial windows (§2.3.2).
        Args:
            x_win_cat: [B, W*C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        b, _, h, w = x_win_cat.shape
        ws = self.spatial_window
        pad_h = (ws - (h % ws)) % ws
        pad_w = (ws - (w % ws)) % ws
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x_win_cat, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_pad = x_win_cat

        hp, wp = x_pad.shape[-2], x_pad.shape[-1]
        nh, nw = hp // ws, wp // ws
        head_ch = self.channels // self.num_heads

        x_norm = self.norm_in(x_pad)
        qkv = self.qkv_dw(self.qkv_conv(x_norm))
        q, k, v = qkv.chunk(3, dim=1)  # Each [B, C, Hp, Wp]

        # Partition into 8x8 local spatial windows using einops.rearrange
        pat_to_win = "b (h_head c_head) (nh wh) (nw ww) -> (b nh nw) h_head c_head (wh ww)"
        q_win = rearrange(
            q, pat_to_win, h_head=self.num_heads, c_head=head_ch, wh=ws, ww=ws
        )
        k_win = rearrange(
            k, pat_to_win, h_head=self.num_heads, c_head=head_ch, wh=ws, ww=ws
        )
        v_win = rearrange(
            v, pat_to_win, h_head=self.num_heads, c_head=head_ch, wh=ws, ww=ws
        )

        q_win = F.normalize(q_win, dim=-1)
        k_win = F.normalize(k_win, dim=-1)

        # Temporal-channel attention across each 8x8 local window
        attn = torch.matmul(q_win, k_win.transpose(-2, -1)) * self.temperature
        attn = torch.softmax(attn, dim=-1)
        out_win = torch.matmul(attn, v_win)

        pat_from_win = "(b nh nw) h_head c_head (wh ww) -> b (h_head c_head) (nh wh) (nw ww)"
        out = rearrange(
            out_win,
            pat_from_win,
            b=b,
            nh=nh,
            nw=nw,
            h_head=self.num_heads,
            c_head=head_ch,
            wh=ws,
            ww=ws,
        )
        return out[:, :, :h, :w]

    def forward(
        self,
        aligned_seq: torch.Tensor,
        sharpness_prior: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            aligned_seq: [B, T, C, H, W]
            sharpness_prior: [B, T, 1, H, W]
        Returns:
            fused_seq: [B, T, C, H, W]
        """
        b, t, c, h, w = aligned_seq.shape
        fused_frames = []

        for idx in range(t):
            win_feat = self._gather_temporal_window(aligned_seq, idx)  # [B, W, C, H, W]
            win_prior = self._gather_temporal_window(sharpness_prior, idx)  # [B, W, 1, H, W]

            # Eq. in §2.3.1: e_tilde_tau = e_tau * (1 + alpha_prior * Softmax(tau_temp * P_tau))
            lucky_weights = torch.softmax(self.prior_temp * win_prior, dim=1)
            win_modulated = win_feat * (1.0 + self.prior_alpha * lucky_weights)

            # Stack W=5 frames along channel dimension: X in R^{B x (5*C) x H/4 x W/4}
            cat_win = rearrange(win_modulated, "b w c h w_sp -> b (w c) h w_sp")
            res = self.reduce_conv(cat_win)

            attn_out = self._windowed_temporal_channel_attention(cat_win)
            x = res + self.proj_out(attn_out)
            x = x + self.ffn(x)
            fused_frames.append(x)

        return torch.stack(fused_frames, dim=1)  # [B, T, C, H, W]


@FUSION_REGISTRY.register("lucky_weighted_avg", "lucky_weighted_average")
class LuckyWeightedAverageFusion(BaseTemporalFusionBlock):
    """
    Classical Lucky Patch weighted fusion baseline (Ablation Variant A/B):
    Weights aligned features strictly by the normalized spectral priority P_t without self-attention.
    """

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.temp = nn.Parameter(torch.tensor(5.0))
        self.refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(
        self,
        aligned_seq: torch.Tensor,
        sharpness_prior: torch.Tensor,
    ) -> torch.Tensor:
        b, t, _, _, _ = aligned_seq.shape
        weights = torch.softmax(self.temp * sharpness_prior, dim=1)  # [B, T, 1, H, W]
        fused_global = (aligned_seq * weights).sum(dim=1, keepdim=True)  # [B, 1, C, H, W]
        mixed = 0.7 * fused_global + 0.3 * aligned_seq
        flat_mixed = rearrange(mixed, "b t c h w -> (b t) c h w")
        return rearrange(self.refine(flat_mixed), "(b t) c h w -> b t c h w", b=b, t=t)
