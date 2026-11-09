"""Strict Configuration Schema Validation (Pydantic V2 + Registry/Signature Introspection)."""

import inspect
from typing import Any, Dict, Literal, Optional, Set

try:
    from pydantic import BaseModel, ConfigDict, Field  # type: ignore

    HAS_PYDANTIC = True
except ImportError:
    BaseModel = object  # type: ignore
    ConfigDict = None  # type: ignore
    Field = None  # type: ignore
    HAS_PYDANTIC = False


ALLOWED_PRIOR_NAMES: Set[str] = {
    "classical_full",
    "cwd_classical_full",
    "minimal_baseline",
    "mean_baseline",
}
ALLOWED_ENCODER_NAMES: Set[str] = {
    "shallow_multiscale",
    "shallow_3scale",
}
ALLOWED_ALIGNMENT_NAMES: Set[str] = {
    "daab_grum_dim",
    "daab_grum_bidirectional",
    "optical_flow_only",
    "flow_only_baseline",
}
ALLOWED_FUSION_NAMES: Set[str] = {
    "mtcsa",
    "mtcsa_spectral_guided",
    "lucky_weighted_avg",
    "lucky_weighted_average",
}
ALLOWED_DECONV_NAMES: Set[str] = {
    "wiener_fft_multichannel",
    "identity_upsample",
}
ALLOWED_DECODER_NAMES: Set[str] = {
    "twin_naf_decoder",
    "single_naf_decoder",
}

# Exact constructor parameter whitelists for each component family (enforced even without torch)
VALID_PRIOR_PARAMS: Dict[str, Set[str]] = {
    "classical_full": {
        "enable_stabilization",
        "stabilize_in_train",
        "use_frf",
        "enable_motion_mask",
        "frf_bandwidth",
        "frf_sigma",
        "frf_patch_size",
        "lambda_s",
        "outlier_weight",
        "motion_window_size",
        "motion_threshold",
        "motion_sharpness_k",
    },
    "cwd_classical_full": {
        "enable_stabilization",
        "stabilize_in_train",
        "use_frf",
        "enable_motion_mask",
        "frf_bandwidth",
        "frf_sigma",
        "frf_patch_size",
        "lambda_s",
        "outlier_weight",
        "motion_window_size",
        "motion_threshold",
        "motion_sharpness_k",
    },
    "minimal_baseline": {"frf_patch_size"},
    "mean_baseline": {"frf_patch_size"},
}

VALID_ENCODER_PARAMS: Set[str] = {"in_channels", "c1", "c2", "c3"}

VALID_ALIGNMENT_PARAMS: Dict[str, Set[str]] = {
    "daab_grum_dim": {
        "channels",
        "num_groups",
        "num_samples",
        "max_offset",
        "num_rdbs",
        "growth_rate",
        "use_checkpoint",
    },
    "daab_grum_bidirectional": {
        "channels",
        "num_groups",
        "num_samples",
        "max_offset",
        "num_rdbs",
        "growth_rate",
        "use_checkpoint",
    },
    "optical_flow_only": {"channels"},
    "flow_only_baseline": {"channels"},
}

VALID_FUSION_PARAMS: Dict[str, Set[str]] = {
    "mtcsa": {"channels", "window_size", "spatial_window", "num_heads"},
    "mtcsa_spectral_guided": {"channels", "window_size", "spatial_window", "num_heads"},
    "lucky_weighted_avg": {"channels"},
    "lucky_weighted_average": {"channels"},
}

VALID_DECONV_PARAMS: Set[str] = {
    "in_channels",
    "out_channels",
    "kernel_size",
    "eps_min",
    "pad_size",
}

VALID_DECODER_PARAMS: Set[str] = {
    "c1",
    "c2",
    "c3",
    "deconv_channels",
    "num_blocks_s2",
    "num_blocks_s1",
    "max_tilt_displacement",
}

VALID_LOSS_PARAMS: Set[str] = {
    "lambda_char",
    "lambda_tilt",
    "lambda_fft",
    "lambda_tv",
    "lambda_cw_ssim",
    "lambda_lpips",
    "phase2_start_ratio",
    "total_iters",
    "charbonnier_eps",
    "use_lpips",
}

VALID_EXPERIMENT_PARAMS: Set[str] = {
    "name",
    "seed",
    "deterministic",
    "output_root",
    "use_wandb",
    "wandb_project",
    "wandb_entity",
    "wandb_group",
    "wandb_run_id",
}


if HAS_PYDANTIC:

    class ExperimentSchema(BaseModel):
        model_config = ConfigDict(extra="forbid")
        name: str = "E_cwdnet_full"
        seed: int = 42
        deterministic: bool = False
        output_root: str = "experiments"
        use_wandb: bool = False
        wandb_project: str = "cwd-net-turbulence"
        wandb_entity: Optional[str] = None
        wandb_group: str = "ablation-study"
        wandb_run_id: Optional[str] = None

    class DataSchema(BaseModel):
        model_config = ConfigDict(extra="forbid")
        dataset_type: str = "atsyn"
        train_root: str = "datasets/ATSyn_dynamic/train"
        val_root: str = "datasets/ATSyn_dynamic/val"
        test_root: str = "datasets/ATSyn_dynamic/test"
        num_frames: int = 12
        crop_size: int = 256
        batch_size: int = 2
        num_workers: int = 4
        allow_synthetic_fallback: bool = True
        fallback_length: int = 64

    class DimsSchema(BaseModel):
        model_config = ConfigDict(extra="forbid")
        c1: int = 64
        c2: int = 96
        c3: int = 128
        deconv_channels: int = 64

    class LossSchema(BaseModel):
        model_config = ConfigDict(extra="forbid")
        lambda_char: float = 0.8
        lambda_tilt: float = 0.2
        lambda_fft: float = 0.05
        lambda_tv: float = 0.01
        lambda_cw_ssim: float = 0.1
        lambda_lpips: float = 0.02
        phase2_start_ratio: float = 0.70
        charbonnier_eps: float = 0.001
        use_lpips: bool = False

    class TrainSchema(BaseModel):
        model_config = ConfigDict(extra="forbid")
        total_iters: int = 200000
        warmup_iters: int = 2000
        lr: float = 0.0002
        min_lr: float = 0.000001
        weight_decay: float = 0.0001
        grad_clip_norm: float = 20.0
        ema_decay: float = 0.999
        use_amp: bool = True
        amp_dtype: Literal["bfloat16", "float16"] = "bfloat16"
        log_interval: int = 50
        val_interval: int = 2000
        save_interval: int = 5000
        keep_last_k_checkpoints: int = 3

    class InferenceSchema(BaseModel):
        model_config = ConfigDict(extra="forbid")
        window_size: int = 12
        temporal_stride: int = 6
        tile_size: int = 512
        tile_overlap: int = 64
        mode: Literal["video", "forensic_snapshot"] = "video"


def _check_unknown_keys(
    section_name: str, given: Dict[str, Any], allowed: Set[str]
) -> None:
    unknown = set(given.keys()) - allowed
    if unknown:
        raise TypeError(
            f"Invalid configuration parameter(s) in '{section_name}': {sorted(unknown)}. "
            f"Allowed parameters: {sorted(allowed)}"
        )


def _validate_live_torch_signatures(model_cfg: Dict[str, Any]) -> None:
    """
    When PyTorch is available, additionally inspects the actual `__init__` signatures
    of the registered classes in `src.models.registry` to guarantee zero TypeError at build time.
    """
    try:
        import src.models.cwd_net  # noqa: F401
        from src.models.registry import (
            ALIGNMENT_REGISTRY,
            DECODER_REGISTRY,
            DECONV_REGISTRY,
            ENCODER_REGISTRY,
            FUSION_REGISTRY,
            PRIOR_REGISTRY,
        )
    except ImportError:
        return

    registry_pairs = [
        ("prior_extractor", PRIOR_REGISTRY),
        ("encoder", ENCODER_REGISTRY),
        ("alignment", ALIGNMENT_REGISTRY),
        ("fusion", FUSION_REGISTRY),
        ("deconv", DECONV_REGISTRY),
        ("decoder", DECODER_REGISTRY),
    ]
    for section_key, reg in registry_pairs:
        sub = dict(model_cfg.get(section_key, {}))
        name = sub.pop("name", None)
        if name is None:
            raise KeyError(f"Missing 'name' key in model.{section_key}")
        cls = reg.get_class(name)
        sig = inspect.signature(cls.__init__)
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not has_var_kw:
            valid_params = set(sig.parameters.keys()) - {"self"}
            _check_unknown_keys(f"model.{section_key} ({cls.__name__})", sub, valid_params)


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strictly validates a resolved configuration dictionary:
      1. Checks top-level sections and schema types (via Pydantic when available + explicit checks).
      2. Checks that all block registry names exist in their respective registries.
      3. Checks that every block sub-dictionary contains only valid constructor arguments.
    Raises `KeyError`, `TypeError`, or `ValueError` immediately if any mismatch is detected.
    """
    allowed_top = {"experiment", "data", "model", "loss", "train", "inference"}
    _check_unknown_keys("root", cfg, allowed_top)

    if HAS_PYDANTIC:
        if "experiment" in cfg:
            ExperimentSchema(**cfg["experiment"])
        if "data" in cfg:
            DataSchema(**cfg["data"])
        if "loss" in cfg:
            LossSchema(**cfg["loss"])
        if "train" in cfg:
            TrainSchema(**cfg["train"])
        if "inference" in cfg:
            InferenceSchema(**cfg["inference"])

    if "experiment" in cfg:
        _check_unknown_keys("experiment", cfg["experiment"], VALID_EXPERIMENT_PARAMS)
    if "loss" in cfg:
        _check_unknown_keys("loss", cfg["loss"], VALID_LOSS_PARAMS)

    model_cfg = cfg.get("model", {})
    allowed_model_keys = {
        "dims",
        "prior_extractor",
        "encoder",
        "alignment",
        "fusion",
        "deconv",
        "decoder",
    }
    _check_unknown_keys("model", model_cfg, allowed_model_keys)

    dims = model_cfg.get("dims", {})
    _check_unknown_keys("model.dims", dims, {"c1", "c2", "c3", "deconv_channels"})

    # Validate Block 0: prior_extractor
    prior_cfg = dict(model_cfg.get("prior_extractor", {}))
    prior_name = prior_cfg.pop("name", None)
    if prior_name not in ALLOWED_PRIOR_NAMES:
        raise KeyError(
            f"Unknown prior_extractor.name={prior_name!r}. Allowed: {sorted(ALLOWED_PRIOR_NAMES)}"
        )
    _check_unknown_keys(
        f"model.prior_extractor[{prior_name}]",
        prior_cfg,
        VALID_PRIOR_PARAMS[prior_name],
    )

    # Validate Block 1.1: encoder
    enc_cfg = dict(model_cfg.get("encoder", {}))
    enc_name = enc_cfg.pop("name", None)
    if enc_name not in ALLOWED_ENCODER_NAMES:
        raise KeyError(
            f"Unknown encoder.name={enc_name!r}. Allowed: {sorted(ALLOWED_ENCODER_NAMES)}"
        )
    _check_unknown_keys(f"model.encoder[{enc_name}]", enc_cfg, VALID_ENCODER_PARAMS)

    # Validate Block 1.2-1.4: alignment
    align_cfg = dict(model_cfg.get("alignment", {}))
    align_name = align_cfg.pop("name", None)
    if align_name not in ALLOWED_ALIGNMENT_NAMES:
        raise KeyError(
            f"Unknown alignment.name={align_name!r}. Allowed: {sorted(ALLOWED_ALIGNMENT_NAMES)}"
        )
    _check_unknown_keys(
        f"model.alignment[{align_name}]",
        align_cfg,
        VALID_ALIGNMENT_PARAMS[align_name],
    )

    # Validate Block 2: fusion
    fusion_cfg = dict(model_cfg.get("fusion", {}))
    fusion_name = fusion_cfg.pop("name", None)
    if fusion_name not in ALLOWED_FUSION_NAMES:
        raise KeyError(
            f"Unknown fusion.name={fusion_name!r}. Allowed: {sorted(ALLOWED_FUSION_NAMES)}"
        )
    _check_unknown_keys(
        f"model.fusion[{fusion_name}]",
        fusion_cfg,
        VALID_FUSION_PARAMS[fusion_name],
    )

    # Validate Block 3: deconv
    deconv_cfg = dict(model_cfg.get("deconv", {}))
    deconv_name = deconv_cfg.pop("name", None)
    if deconv_name not in ALLOWED_DECONV_NAMES:
        raise KeyError(
            f"Unknown deconv.name={deconv_name!r}. Allowed: {sorted(ALLOWED_DECONV_NAMES)}"
        )
    _check_unknown_keys(
        f"model.deconv[{deconv_name}]", deconv_cfg, VALID_DECONV_PARAMS
    )

    # Validate Block 4: decoder
    dec_cfg = dict(model_cfg.get("decoder", {}))
    dec_name = dec_cfg.pop("name", None)
    if dec_name not in ALLOWED_DECODER_NAMES:
        raise KeyError(
            f"Unknown decoder.name={dec_name!r}. Allowed: {sorted(ALLOWED_DECODER_NAMES)}"
        )
    _check_unknown_keys(f"model.decoder[{dec_name}]", dec_cfg, VALID_DECODER_PARAMS)

    # Live signature introspection when torch is available
    _validate_live_torch_signatures(model_cfg)
    return cfg
