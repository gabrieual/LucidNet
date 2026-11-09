"""Typed tensor containers decoupling training supervision from inference."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import torch


@dataclass
class VideoBatch:
    """
    Input data contract (Interface Segregation Principle).
    During inference, only `degraded` is required.
    During synthetic training (ATSyn / P2S), `gt` and `tilt_only` are provided.
    """

    degraded: torch.Tensor  # Shape: [B, T, C, H, W], float32 in [0, 1]
    gt: Optional[torch.Tensor] = None  # Shape: [B, T, C, H, W], float32 in [0, 1]
    tilt_only: Optional[torch.Tensor] = None  # Shape: [B, T, C, H, W], float32 in [0, 1]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device, non_blocking: bool = True) -> "VideoBatch":
        """Move all available tensors in the batch to the target device."""
        return VideoBatch(
            degraded=self.degraded.to(device, non_blocking=non_blocking),
            gt=self.gt.to(device, non_blocking=non_blocking) if self.gt is not None else None,
            tilt_only=(
                self.tilt_only.to(device, non_blocking=non_blocking)
                if self.tilt_only is not None
                else None
            ),
            meta=self.meta,
        )


@dataclass
class PriorOutput:
    """Output contract of Block 0 (Classical Physics-Statistical Prior Extractor)."""

    stabilized_video: torch.Tensor  # [B, T, C, H, W]
    reference_frames: torch.Tensor  # [B, T, C, H, W] (FRF mixed with dynamic mask M_t)
    frf_static: torch.Tensor  # [B, C, H, W] (Pure Frequency-Aware Reference Frame)
    motion_mask: torch.Tensor  # [B, T, 1, H, W] in [0, 1]
    sharpness_prior: torch.Tensor  # [B, T, 1, H//4, W//4] (P_t priority map for MTCSA)
    cn2_estimate: torch.Tensor  # [B] scalar turbulence strength estimate per clip


@dataclass
class ModelOutput:
    """Output contract of the complete restoration network."""

    restored: torch.Tensor  # [B, T, C, H, W] final restored video J_hat
    rectified_rgb: torch.Tensor  # [B, T, C, H, W] tilt-rectified input I_rect
    inverse_tilt_field: torch.Tensor  # [B, T, 2, H, W] predicted inverse tilt T_hat^{-1}
    frf_reference: torch.Tensor  # [B, C, H, W] static FRF reference frame
    aux: Dict[str, Any] = field(default_factory=dict)  # PSFs K_hat, gamma, beta_deconv, etc.
