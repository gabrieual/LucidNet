"""Abstract Base Classes enforcing Liskov Substitution & Interface Segregation across all blocks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

from src.contracts.batch import ModelOutput, PriorOutput, VideoBatch


class BasePriorExtractor(nn.Module, ABC):
    """Contract for Block 0: Classical Physics-Statistical Prior Extractor."""

    @abstractmethod
    def estimate_global_cn2(self, video: torch.Tensor) -> torch.Tensor:
        """
        Computes the global turbulence strength Cn^2 [B] across the full-resolution frame
        before spatial tiling in production inference (§9.2 of arquitetura.md).
        """

    @abstractmethod
    def forward(
        self,
        video: torch.Tensor,
        precomputed_cn2: Optional[torch.Tensor] = None,
    ) -> PriorOutput:
        """
        Args:
            video: Degraded input sequence [B, T, C, H, W]
            precomputed_cn2: Optional precomputed scene-wide Cn^2 [B] (used in spatial tiling)
        Returns:
            PriorOutput containing stabilized frames, FRF reference, motion mask, and P_t prior.
        """


class BaseMultiScaleEncoder(nn.Module, ABC):
    """Contract for Block 1.1: Shared shallow multi-scale feature extractor."""

    @abstractmethod
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Frame tensor [N, C, H, W]
        Returns:
            f1: [N, C1, H, W]       (Scale 1/1, C1=64)
            f2: [N, C2, H//2, W//2] (Scale 1/2, C2=96)
            f3: [N, C3, H//4, W//4] (Scale 1/4, C3=128)
        """


class BaseAlignmentBlock(nn.Module, ABC):
    """Contract for Block 1.2-1.4: Frame-to-Reference Registration & Recurrent Update."""

    @abstractmethod
    def forward(
        self,
        f3_seq: torch.Tensor,
        ref_f3_seq: torch.Tensor,
        frames_low: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            f3_seq: [B, T, C3, H//4, W//4] deep features of input frames
            ref_f3_seq: [B, T, C3, H//4, W//4] deep features of initial reference frames
            frames_low: [B, T, 3, H//4, W//4] downsampled RGB frames for optical flow
        Returns:
            aligned_seq: [B, T, C3, H//4, W//4] bidirectionally aligned & integrated embeddings
        """


class BaseTemporalFusionBlock(nn.Module, ABC):
    """Contract for Block 2: Temporal-Channel Lucky Fusion (MTCSA)."""

    @abstractmethod
    def forward(
        self,
        aligned_seq: torch.Tensor,
        sharpness_prior: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            aligned_seq: [B, T, C3, H//4, W//4]
            sharpness_prior: [B, T, 1, H//4, W//4] classical priority map P_t
        Returns:
            fused_seq: [B, T, C3, H//4, W//4]
        """


class BaseDeconvBlock(nn.Module, ABC):
    """Contract for Block 3: Blind Wiener Deconvolution in 2D-FFT (or Identity for ablation)."""

    @abstractmethod
    def forward(
        self, f_lucky: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            f_lucky: [B*T, C3, H//4, W//4] temporally fused features (C3=128)
        Returns:
            f_deconv: [B*T, C_deconv, H//2, W//2] upsampled and deconvolved features (C_deconv=64)
            aux_dict: Dictionary with estimated PSFs 'psf_kernels', 'snr_gamma', 'beta_deconv'
        """


class BaseTwinDecoder(nn.Module, ABC):
    """Contract for Block 4: Twin Decoder (Tilt Rectifier Branch A + NAFNet Branch B)."""

    @abstractmethod
    def forward(
        self,
        frames: torch.Tensor,
        f1_seq: torch.Tensor,
        f2_seq: torch.Tensor,
        f_lucky_seq: torch.Tensor,
        f_deconv_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            frames: [B, T, 3, H, W] stabilized input frames
            f1_seq: [B, T, C1, H, W] shallow features scale 1/1 (C1=64)
            f2_seq: [B, T, C2, H//2, W//2] shallow features scale 1/2 (C2=96)
            f_lucky_seq: [B, T, C3, H//4, W//4] fused features scale 1/4 (C3=128)
            f_deconv_seq: [B, T, C_deconv, H//2, W//2] Wiener-deconvolved features scale 1/2 (C_deconv=64)
        Returns:
            restored: [B, T, 3, H, W] final restored frames J_hat
            rectified_rgb: [B, T, 3, H, W] tilt-rectified input frames
            inv_tilt_field: [B, T, 2, H, W] predicted inverse tilt fields in pixel displacement
        """


class InferenceModel(nn.Module, ABC):
    """
    Interface Segregation Principle (§4.4 of arquitetura.md):
    Pure inference contract requiring only the degraded video tensor (and optional global Cn^2),
    decoupled from ground-truth or auxiliary training targets.
    """

    @abstractmethod
    def forward(
        self,
        degraded_video: torch.Tensor,
        precomputed_cn2: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """Runs restoration inference on `degraded_video` [B, T, C, H, W]."""


class TrainingLossWrapper(nn.Module, ABC):
    """
    Interface Segregation Principle (§4.4 of arquitetura.md):
    Contract for computing training losses from `ModelOutput` and `VideoBatch`
    (which carries `gt` and `tilt_only`).
    """

    @abstractmethod
    def forward(
        self,
        output: ModelOutput,
        batch: VideoBatch,
        current_iter: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes composite curriculum loss and scalar logging dictionary."""
