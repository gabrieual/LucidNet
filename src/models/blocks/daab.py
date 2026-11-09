"""Block 1.2-1.4: Deformable Attention Alignment Block (DAAB) & Recurrent Alignment Stage."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.contracts.interfaces import BaseAlignmentBlock
from src.models.registry import ALIGNMENT_REGISTRY
from src.models.blocks.rdb_dim import ResidualDenseBlock, DeepIntegrationModule
from src.models.blocks.grum import GatedReferenceUpdateModule
from src.utils.tensor_ops import rearrange


def flow_warp(
    x: torch.Tensor,
    flow: torch.Tensor,
    interpolation: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Warps tensor `x` [N, C, H, W] using pixel displacement field `flow` [N, 2, H, W]
    where flow[:, 0] is horizontal shift dx and flow[:, 1] is vertical shift dy.

    CRITICAL PRECISION GUARD: The coordinate meshgrid and `F.grid_sample` are always
    computed in `float32` so sub-pixel displacements are never quantized to integer pixels
    when running under `bfloat16` or `float16` AMP.
    """
    orig_dtype = x.dtype
    n, c, h, w = x.shape
    flow_f32 = flow.float()

    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=x.device, dtype=torch.float32),
        torch.arange(0, w, device=x.device, dtype=torch.float32),
        indexing="ij",
    )
    grid_x = grid_x.unsqueeze(0) + flow_f32[:, 0]
    grid_y = grid_y.unsqueeze(0) + flow_f32[:, 1]

    norm_x = 2.0 * grid_x / max(w - 1, 1) - 1.0
    norm_y = 2.0 * grid_y / max(h - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)  # [N, H, W, 2] in float32

    warped = F.grid_sample(
        x.float(),
        grid,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return warped.to(orig_dtype)


class LightweightPyramidFlow(nn.Module):
    """
    Fast 2-level coarse-to-fine optical flow estimator (SPyNet-lite style)
    operating on downsampled frames at scale 1/4 (Line 3 of Algorithm 1 in DATUM).
    """

    def __init__(self, hidden_ch: int = 32) -> None:
        super().__init__()
        self.coarse_net = nn.Sequential(
            nn.Conv2d(6, hidden_ch, kernel_size=5, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_ch, 2, kernel_size=3, padding=1),
        )
        self.fine_net = nn.Sequential(
            nn.Conv2d(8, hidden_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_ch, 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.coarse_net[-1].weight)
        nn.init.zeros_(self.coarse_net[-1].bias)
        nn.init.zeros_(self.fine_net[-1].weight)
        nn.init.zeros_(self.fine_net[-1].bias)

    def forward(
        self, img_target: torch.Tensor, img_source: torch.Tensor
    ) -> torch.Tensor:
        """Estimate displacement field aligning img_source -> img_target."""
        t_half = F.avg_pool2d(img_target, kernel_size=2, stride=2)
        s_half = F.avg_pool2d(img_source, kernel_size=2, stride=2)

        flow_coarse = self.coarse_net(torch.cat([t_half, s_half], dim=1))
        flow_up = (
            F.interpolate(
                flow_coarse,
                size=img_target.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            * 2.0
        )

        s_warped = flow_warp(img_source, flow_up)
        flow_res = self.fine_net(torch.cat([img_target, s_warped, flow_up], dim=1))
        return flow_up + flow_res


class DeformableAttentionAlignmentBlock(nn.Module):
    """
    Implements Algorithm 1 (DAAB) of DATUM (Zhang et al., CVPR 2024) & §2.2.2 of `arquitetura.md`:
    Registers current frame feature f_t^3 to hidden reference feature r_{t-1}
    using multi-group (G=8) multi-sampling deformable cross-attention.
    """

    def __init__(
        self,
        channels: int = 128,
        num_groups: int = 8,
        num_samples: int = 4,
        max_offset: float = 6.0,
        growth_rate: int = 32,
    ) -> None:
        super().__init__()
        assert channels % num_groups == 0, "channels must be divisible by num_groups"
        self.channels = channels
        self.num_groups = num_groups
        self.group_ch = channels // num_groups
        self.num_samples = num_samples
        self.max_offset = max_offset

        # Offset predictor from Concat(f_hat_t, r_{t-1}, O_hat^{f->r})
        self.offset_rdb = ResidualDenseBlock(
            channels * 2 + 2, growth_rate=growth_rate, num_layers=3
        )
        self.offset_head = nn.Conv2d(
            channels * 2 + 2,
            num_groups * num_samples * 2,
            kernel_size=3,
            padding=1,
        )
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

        self.proj_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_k = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
        )

    def forward(
        self,
        f_curr: torch.Tensor,
        r_prev: torch.Tensor,
        rough_flow: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_curr: [B, C, H, W] current frame feature f_t^3
            r_prev: [B, C, H, W] reference feature r_{t-1}
            rough_flow: [B, 2, H, W] pre-alignment flow O_hat_t^{f->r}
        Returns:
            f_aligned: [B, C, H, W] registered feature
            updated_flow: [B, 2, H, W] refined alignment flow O_t^{f->r}
        """
        b, c, h, w = f_curr.shape

        # Line 5 of Algorithm 1: Pre-align f_curr using rough deformation field
        f_pre = flow_warp(f_curr, rough_flow)

        # Lines 7-8 of Algorithm 1: Predict group offsets Delta_o_t^(g)
        feat_cat = self.offset_rdb(torch.cat([f_pre, r_prev, rough_flow], dim=1))
        delta_offsets = torch.tanh(self.offset_head(feat_cat)) * self.max_offset
        delta_offsets = delta_offsets.view(
            b, self.num_groups, self.num_samples, 2, h, w
        )
        total_flows = (
            rough_flow.unsqueeze(1).unsqueeze(2) + delta_offsets
        )  # [B, G, S, 2, H, W]

        # Lines 9-10 of Algorithm 1: Multi-group deformable attention
        q = self.proj_q(r_prev).view(b, self.num_groups, self.group_ch, h, w)
        k_base = self.proj_k(f_curr).view(b, self.num_groups, self.group_ch, h, w)
        v_base = self.proj_v(f_curr).view(b, self.num_groups, self.group_ch, h, w)

        k_samples, v_samples = [], []
        k_flat = k_base.reshape(b * self.num_groups, self.group_ch, h, w)
        v_flat = v_base.reshape(b * self.num_groups, self.group_ch, h, w)

        for s_idx in range(self.num_samples):
            flow_s = total_flows[:, :, s_idx].reshape(b * self.num_groups, 2, h, w)
            k_s = flow_warp(k_flat, flow_s).view(
                b, self.num_groups, self.group_ch, h, w
            )
            v_s = flow_warp(v_flat, flow_s).view(
                b, self.num_groups, self.group_ch, h, w
            )
            k_samples.append(k_s)
            v_samples.append(v_s)

        k_stack = torch.stack(k_samples, dim=2)  # [B, G, S, C_g, H, W]
        v_stack = torch.stack(v_samples, dim=2)  # [B, G, S, C_g, H, W]

        scale = float(self.group_ch) ** -0.5
        attn_logits = (q.unsqueeze(2) * k_stack).sum(dim=3) * scale  # [B, G, S, H, W]
        attn_weights = torch.softmax(attn_logits, dim=2).unsqueeze(3)

        aligned_groups = (attn_weights * v_stack).sum(dim=2)  # [B, G, C_g, H, W]
        f_aligned = self.proj_out(aligned_groups.reshape(b, c, h, w))
        f_aligned = f_aligned + self.ffn(f_aligned)

        updated_flow = total_flows.mean(dim=(1, 2))  # [B, 2, H, W]
        return f_aligned, updated_flow


@ALIGNMENT_REGISTRY.register("daab_grum_dim", "daab_grum_bidirectional")
class RecurrentReferenceAlignmentStage(BaseAlignmentBlock):
    """
    Full Block 1.2-1.4 of CWD-Net (§2.2.2 - §2.2.3 of `arquitetura.md`):
    Performs bidirectional recurrent Frame-to-Reference registration using:
      - LightweightPyramidFlow
      - DeformableAttentionAlignmentBlock (DAAB)
      - DeepIntegrationModule (DIM: 4 to 6 Residual Dense Blocks)
      - GatedReferenceUpdateModule (GRUM: ConvGRU Bidirecional)
    Initialized with the classical FRF reference features from Block 0.
    """

    def __init__(
        self,
        channels: int = 128,
        num_groups: int = 8,
        num_samples: int = 4,
        max_offset: float = 6.0,
        num_rdbs: int = 4,
        growth_rate: int = 32,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.flow_net = LightweightPyramidFlow()
        self.daab = DeformableAttentionAlignmentBlock(
            channels=channels,
            num_groups=num_groups,
            num_samples=num_samples,
            max_offset=max_offset,
            growth_rate=growth_rate,
        )
        self.dim = DeepIntegrationModule(
            in_channels=channels * 3,
            embed_channels=channels,
            num_rdbs=num_rdbs,
            growth_rate=growth_rate,
            use_checkpoint=use_checkpoint,
        )
        self.grum = GatedReferenceUpdateModule(channels=channels)
        self.bi_fuse = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def _propagate_direction(
        self,
        f3_seq: torch.Tensor,
        ref_f3_seq: torch.Tensor,
        frames_low: torch.Tensor,
    ) -> torch.Tensor:
        b, t, c, h, w = f3_seq.shape
        r_prev = ref_f3_seq[:, 0]
        flow_prev = torch.zeros(
            (b, 2, h, w), device=f3_seq.device, dtype=f3_seq.dtype
        )

        embeddings = []
        for idx in range(t):
            f_curr = f3_seq[:, idx]
            r_guided = 0.7 * r_prev + 0.3 * ref_f3_seq[:, idx]

            if idx == 0:
                rough_flow = flow_prev
            else:
                flow_inter = self.flow_net(
                    frames_low[:, idx - 1], frames_low[:, idx]
                )
                rough_flow = flow_prev + flow_warp(flow_inter, flow_prev)

            f_reg, flow_prev = self.daab(f_curr, r_guided, rough_flow)
            e_curr = self.dim(f_reg, r_guided, f_curr)
            r_prev = self.grum(r_guided, e_curr)
            embeddings.append(e_curr)

        return torch.stack(embeddings, dim=1)  # [B, T, C, H, W]

    def forward(
        self,
        f3_seq: torch.Tensor,
        ref_f3_seq: torch.Tensor,
        frames_low: torch.Tensor,
    ) -> torch.Tensor:
        e_fw = self._propagate_direction(f3_seq, ref_f3_seq, frames_low)
        e_bw = self._propagate_direction(
            f3_seq.flip(dims=[1]),
            ref_f3_seq.flip(dims=[1]),
            frames_low.flip(dims=[1]),
        ).flip(dims=[1])

        b, t, _, _, _ = e_fw.shape
        cat_bi = rearrange(
            torch.cat([e_fw, e_bw], dim=2), "b t c h w -> (b t) c h w"
        )
        fused = self.bi_fuse(cat_bi)
        return rearrange(fused, "(b t) c h w -> b t c h w", b=b, t=t)


@ALIGNMENT_REGISTRY.register("optical_flow_only", "flow_only_baseline")
class OpticalFlowOnlyAlignmentStage(BaseAlignmentBlock):
    """
    Baseline alignment stage (for Ablation Variant A/B):
    Warps features directly to the reference frame without Deformable Attention or ConvGRU.
    """

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.flow_net = LightweightPyramidFlow()
        self.proj = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)

    def forward(
        self,
        f3_seq: torch.Tensor,
        ref_f3_seq: torch.Tensor,
        frames_low: torch.Tensor,
    ) -> torch.Tensor:
        b, t, _, _, _ = f3_seq.shape
        ref_low = frames_low.mean(dim=1, keepdim=True).expand_as(frames_low)
        flow = self.flow_net(
            rearrange(ref_low, "b t c h w -> (b t) c h w"),
            rearrange(frames_low, "b t c h w -> (b t) c h w"),
        )
        warped = flow_warp(rearrange(f3_seq, "b t c h w -> (b t) c h w"), flow)
        ref_flat = rearrange(ref_f3_seq, "b t c h w -> (b t) c h w")
        out = self.proj(torch.cat([warped, ref_flat], dim=1))
        return rearrange(out, "(b t) c h w -> b t c h w", b=b, t=t)
