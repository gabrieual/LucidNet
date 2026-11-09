"""Orchestrator of Block 0 registered in PRIOR_REGISTRY."""

from typing import Optional
import torch
from src.contracts.batch import PriorOutput
from src.contracts.interfaces import BasePriorExtractor
from src.models.registry import PRIOR_REGISTRY
from src.models.classical.stabilization import FFTGlobalStabilizer
from src.models.classical.frf_reference import (
    FrequencyAwareReferenceBuilder,
    MeanReferenceBuilder,
)
from src.models.classical.motion_cn2 import MotionCn2Estimator
from src.models.classical.spectral_prior import SpectralSharpnessPrior


@PRIOR_REGISTRY.register("classical_full", "cwd_classical_full")
class ClassicalPriorExtractor(BasePriorExtractor):
    """
    Full Block 0 of CWD-Net (§2.1 of `arquitetura.md`):
      1. 2D-FFT Global Stabilization (xcorr2d anchored at zero-mean centroid)
      2. Frequency-Aware Reference Frame (FRF from CDSP, h=0.05, sigma=0.5)
      3. Cn^2 turbulence strength + AOF Motion Segmentation Mask (M_t)
      4. 6-Direction Complex Gabor/Wavelet + 8x8 FFT Bandpass * Outlier Similarity Priority Map (P_t)
    """

    def __init__(
        self,
        enable_stabilization: bool = True,
        stabilize_in_train: bool = False,
        use_frf: bool = True,
        enable_motion_mask: bool = True,
        frf_bandwidth: float = 0.05,
        frf_sigma: float = 0.5,
        frf_patch_size: int = 8,
        lambda_s: float = 0.02,
        outlier_weight: float = 1.0,
        motion_window_size: int = 5,
        motion_threshold: float = 0.08,
        motion_sharpness_k: float = 25.0,
    ) -> None:
        super().__init__()
        self.stabilizer = FFTGlobalStabilizer(
            enabled=enable_stabilization,
            stabilize_in_train=stabilize_in_train,
        )
        self.ref_builder = (
            FrequencyAwareReferenceBuilder(bandwidth_h=frf_bandwidth, sigma=frf_sigma)
            if use_frf
            else MeanReferenceBuilder()
        )
        self.enable_motion_mask = enable_motion_mask
        self.motion_cn2 = MotionCn2Estimator(
            window_size=motion_window_size,
            motion_threshold=motion_threshold,
            sharpness_k=motion_sharpness_k,
        )
        self.spectral_prior = SpectralSharpnessPrior(
            patch_size=frf_patch_size,
            lambda_s=lambda_s,
            outlier_weight=outlier_weight,
        )

    def estimate_global_cn2(self, video: torch.Tensor) -> torch.Tensor:
        return self.motion_cn2.estimate_cn2(video)

    def forward(
        self,
        video: torch.Tensor,
        precomputed_cn2: Optional[torch.Tensor] = None,
    ) -> PriorOutput:
        stabilized = self.stabilizer(video)
        frf_static = self.ref_builder(stabilized)  # [B, C, H, W]

        if self.enable_motion_mask:
            cn2_est, motion_mask = self.motion_cn2(
                stabilized, frf_static, precomputed_cn2=precomputed_cn2
            )
        else:
            b, t, _, h, w = stabilized.shape
            cn2_est = (
                precomputed_cn2
                if precomputed_cn2 is not None
                else torch.zeros(b, device=video.device, dtype=video.dtype)
            )
            motion_mask = torch.zeros(
                (b, t, 1, h, w), device=video.device, dtype=video.dtype
            )

        # Blend static FRF with current frame in dynamic regions (Eq. in §2.1.3)
        frf_expanded = frf_static.unsqueeze(1).expand_as(stabilized)
        reference_frames = (1.0 - motion_mask) * frf_expanded + motion_mask * stabilized

        sharpness_prior = self.spectral_prior(
            stabilized, reference_frames, target_scale=4
        )

        return PriorOutput(
            stabilized_video=stabilized,
            reference_frames=reference_frames,
            frf_static=frf_static,
            motion_mask=motion_mask,
            sharpness_prior=sharpness_prior,
            cn2_estimate=cn2_est,
        )


@PRIOR_REGISTRY.register("minimal_baseline", "mean_baseline")
class MinimalPriorExtractor(BasePriorExtractor):
    """
    Baseline Block 0 corresponding to Variant A (proposta.pdf):
    Uses simple temporal mean reference I_bar = (1/T)*sum(I_t) and 2D-FFT/Gabor sharpness without FRF.
    """

    def __init__(self, frf_patch_size: int = 8) -> None:
        super().__init__()
        self.ref_builder = MeanReferenceBuilder()
        self.spectral_prior = SpectralSharpnessPrior(
            patch_size=frf_patch_size, outlier_weight=0.0
        )

    def estimate_global_cn2(self, video: torch.Tensor) -> torch.Tensor:
        return torch.zeros(video.shape[0], device=video.device, dtype=video.dtype)

    def forward(
        self,
        video: torch.Tensor,
        precomputed_cn2: Optional[torch.Tensor] = None,
    ) -> PriorOutput:
        b, t, _, h, w = video.shape
        mean_ref = self.ref_builder(video)
        ref_frames = mean_ref.unsqueeze(1).expand_as(video)
        sharpness_prior = self.spectral_prior(video, ref_frames, target_scale=4)
        cn2 = (
            precomputed_cn2
            if precomputed_cn2 is not None
            else torch.zeros(b, device=video.device, dtype=video.dtype)
        )
        return PriorOutput(
            stabilized_video=video,
            reference_frames=ref_frames,
            frf_static=mean_ref,
            motion_mask=torch.zeros(
                (b, t, 1, h, w), device=video.device, dtype=video.dtype
            ),
            sharpness_prior=sharpness_prior,
            cn2_estimate=cn2,
        )
