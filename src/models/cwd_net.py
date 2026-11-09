"""CWD-Net Top-Level Orchestrator enforcing Dependency Inversion & Interface Segregation."""

from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F

from src.contracts.batch import ModelOutput, PriorOutput
from src.contracts.interfaces import (
    BaseAlignmentBlock,
    BaseDeconvBlock,
    BaseMultiScaleEncoder,
    BasePriorExtractor,
    BaseTemporalFusionBlock,
    BaseTwinDecoder,
    InferenceModel,
)
from src.models.registry import (
    ALIGNMENT_REGISTRY,
    DECODER_REGISTRY,
    DECONV_REGISTRY,
    ENCODER_REGISTRY,
    FUSION_REGISTRY,
    PRIOR_REGISTRY,
)
from src.utils.tensor_ops import rearrange
import src.models.classical  # noqa: F401 - register classical prior extractors
import src.models.blocks  # noqa: F401 - register neural blocks


class CWDNet(InferenceModel):
    """
    CLEAR-Wiener-DATUM Network (CWD-Net).
    Implements `InferenceModel` (§4.4) and depends solely on abstract block contracts
    injected at construction time (§4.5 Dependency Inversion Principle).
    """

    def __init__(
        self,
        prior_extractor: BasePriorExtractor,
        encoder: BaseMultiScaleEncoder,
        alignment: BaseAlignmentBlock,
        fusion: BaseTemporalFusionBlock,
        deconv: BaseDeconvBlock,
        decoder: BaseTwinDecoder,
    ) -> None:
        super().__init__()
        self.prior_extractor = prior_extractor
        self.encoder = encoder
        self.alignment = alignment
        self.fusion = fusion
        self.deconv = deconv
        self.decoder = decoder

    @torch.no_grad()
    def estimate_global_cn2(self, degraded_video: torch.Tensor) -> torch.Tensor:
        """Computes scene-wide Cn^2 [B] once on the full-resolution clip before spatial tiling."""
        return self.prior_extractor.estimate_global_cn2(degraded_video)

    def forward(
        self,
        degraded_video: torch.Tensor,
        precomputed_cn2: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        Args:
            degraded_video: [B, T, C, H, W] input video clip in [0, 1]
            precomputed_cn2: Optional scene-wide Cn^2 [B] computed prior to spatial tiling
        Returns:
            ModelOutput containing restored video, rectified_rgb, inverse_tilt_field,
            frf_reference, and auxiliary diagnostics (PSF kernels, SNR gamma, beta_deconv, Cn2).
        """
        b, t, _, h, w = degraded_video.shape
        h4, w4 = h // 4, w // 4

        # Block 0: Classical Physics-Statistical Prior Extraction (No trainable params)
        with torch.no_grad():
            priors: PriorOutput = self.prior_extractor(
                degraded_video, precomputed_cn2=precomputed_cn2
            )

        stab_frames = priors.stabilized_video  # [B, T, C, H, W]
        ref_frames = priors.reference_frames  # [B, T, C, H, W]

        # Block 1.1: Shared Multi-Scale Feature Extraction
        f1_flat, f2_flat, f3_flat = self.encoder(
            rearrange(stab_frames, "b t c h w -> (b t) c h w")
        )
        _, _, ref_f3_flat = self.encoder(
            rearrange(ref_frames, "b t c h w -> (b t) c h w")
        )

        f1_seq = rearrange(f1_flat, "(b t) c h w -> b t c h w", b=b, t=t)
        f2_seq = rearrange(f2_flat, "(b t) c h w -> b t c h w", b=b, t=t)
        f3_seq = rearrange(f3_flat, "(b t) c h w -> b t c h w", b=b, t=t)
        ref_f3_seq = rearrange(ref_f3_flat, "(b t) c h w -> b t c h w", b=b, t=t)

        # Downsampled RGB frames at 1/4 resolution for coarse optical flow prior
        frames_low_flat = F.interpolate(
            rearrange(stab_frames, "b t c h w -> (b t) c h w"),
            size=(h4, w4),
            mode="bilinear",
            align_corners=False,
        )
        frames_low = rearrange(frames_low_flat, "(b t) c h w -> b t c h w", b=b, t=t)

        # Block 1.2 - 1.4: Recurrent Frame-to-Reference Deformable Registration (Scale 1/4)
        aligned_seq = self.alignment(f3_seq, ref_f3_seq, frames_low)

        # Block 2: Multi-Head Temporal-Channel Lucky Fusion guided by P_t (Scale 1/4)
        f_lucky_seq = self.fusion(aligned_seq, priors.sharpness_prior)

        # Block 3: Multi-Channel Blind Wiener Deconvolution in 2D-FFT (1/4 -> 1/2, M=64 ch)
        f_deconv_flat, deconv_aux = self.deconv(
            rearrange(f_lucky_seq, "b t c h w -> (b t) c h w")
        )
        f_deconv_seq = rearrange(f_deconv_flat, "(b t) c h w -> b t c h w", b=b, t=t)

        # Block 4: Twin Decoder (Tilt Rectifier Branch A + NAFNet Restoration Branch B)
        restored, rectified_rgb, inv_tilt = self.decoder(
            frames=stab_frames,
            f1_seq=f1_seq,
            f2_seq=f2_seq,
            f_lucky_seq=f_lucky_seq,
            f_deconv_seq=f_deconv_seq,
        )

        aux: Dict[str, Any] = {
            **deconv_aux,
            "cn2_estimate": priors.cn2_estimate,
            "motion_mask": priors.motion_mask,
            "sharpness_prior": priors.sharpness_prior,
        }
        return ModelOutput(
            restored=restored,
            rectified_rgb=rectified_rgb,
            inverse_tilt_field=inv_tilt,
            frf_reference=priors.frf_static,
            aux=aux,
        )


def build_cwd_net(config: Dict[str, Any]) -> CWDNet:
    """
    Factory function assembling CWDNet (or any ablation variant A-E) from a validated config dict.
    """
    model_cfg = config.get("model", config)
    dims = model_cfg.get(
        "dims", {"c1": 64, "c2": 96, "c3": 128, "deconv_channels": 64}
    )
    c1 = int(dims.get("c1", 64))
    c2 = int(dims.get("c2", 96))
    c3 = int(dims.get("c3", 128))
    deconv_ch = int(dims.get("deconv_channels", 64))

    prior_cfg = dict(model_cfg.get("prior_extractor", {"name": "classical_full"}))
    prior_name = prior_cfg.pop("name")
    prior_extractor = PRIOR_REGISTRY.build(prior_name, **prior_cfg)

    enc_cfg = dict(model_cfg.get("encoder", {"name": "shallow_multiscale"}))
    enc_name = enc_cfg.pop("name")
    enc_cfg.setdefault("c1", c1)
    enc_cfg.setdefault("c2", c2)
    enc_cfg.setdefault("c3", c3)
    encoder = ENCODER_REGISTRY.build(enc_name, **enc_cfg)

    align_cfg = dict(model_cfg.get("alignment", {"name": "daab_grum_dim"}))
    align_name = align_cfg.pop("name")
    align_cfg.setdefault("channels", c3)
    alignment = ALIGNMENT_REGISTRY.build(align_name, **align_cfg)

    fusion_cfg = dict(model_cfg.get("fusion", {"name": "mtcsa"}))
    fusion_name = fusion_cfg.pop("name")
    fusion_cfg.setdefault("channels", c3)
    fusion = FUSION_REGISTRY.build(fusion_name, **fusion_cfg)

    deconv_cfg = dict(model_cfg.get("deconv", {"name": "wiener_fft_multichannel"}))
    deconv_name = deconv_cfg.pop("name")
    deconv_cfg.setdefault("in_channels", c3)
    deconv_cfg.setdefault("out_channels", deconv_ch)
    deconv = DECONV_REGISTRY.build(deconv_name, **deconv_cfg)

    dec_cfg = dict(model_cfg.get("decoder", {"name": "twin_naf_decoder"}))
    dec_name = dec_cfg.pop("name")
    dec_cfg.setdefault("c1", c1)
    dec_cfg.setdefault("c2", c2)
    dec_cfg.setdefault("c3", c3)
    dec_cfg.setdefault("deconv_channels", deconv_ch)
    decoder = DECODER_REGISTRY.build(dec_name, **dec_cfg)

    return CWDNet(
        prior_extractor=prior_extractor,
        encoder=encoder,
        alignment=alignment,
        fusion=fusion,
        deconv=deconv,
        decoder=decoder,
    )
