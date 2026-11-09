"""Unit tests for YAML configurations, Schema Validation, AST Signature Alignment & Liskov Contracts."""

import ast
from pathlib import Path
from typing import Dict, Set, Tuple
import pytest

from src.utils.config_loader import load_config
from src.utils.config_schema import (
    VALID_ALIGNMENT_PARAMS,
    VALID_DECODER_PARAMS,
    VALID_DECONV_PARAMS,
    VALID_ENCODER_PARAMS,
    VALID_FUSION_PARAMS,
    VALID_LOSS_PARAMS,
    VALID_PRIOR_PARAMS,
    validate_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ABLATION_CONFIGS = [
    "configs/ablations/A_baseline_proposta.yaml",
    "configs/ablations/B_frf_only.yaml",
    "configs/ablations/C_daab_mtcsa_no_wiener.yaml",
    "configs/ablations/D_twin_decoder_no_wiener.yaml",
    "configs/ablations/E_cwdnet_full.yaml",
]


def _extract_registered_classes_and_init_args(
    py_path: Path,
) -> Dict[str, Tuple[str, Set[str]]]:
    """
    Parses a Python file's AST to extract all `@*_REGISTRY.register(...)` decorator names
    and the exact `__init__` parameter names of the decorated class.
    Returns `{registered_name: (class_name, set_of_init_param_names)}`.
    """
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    results: Dict[str, Tuple[str, Set[str]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            reg_names = []
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "register"
                ):
                    for arg in dec.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            reg_names.append(arg.value)
            if not reg_names:
                continue
            init_params: Set[str] = set()
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    for a in item.args.args:
                        if a.arg != "self":
                            init_params.add(a.arg)
            for rname in reg_names:
                results[rname] = (node.name, init_params)
    return results


@pytest.mark.parametrize("cfg_rel_path", ABLATION_CONFIGS)
def test_yaml_inheritance_and_strict_schema(cfg_rel_path: str) -> None:
    """Verifies that all 5 ablation YAML configs resolve `_base_` and pass strict schema validation."""
    cfg = load_config(PROJECT_ROOT / cfg_rel_path, validate=True)
    assert cfg["model"]["dims"]["c1"] == 64
    assert cfg["model"]["dims"]["c2"] == 96
    assert cfg["model"]["dims"]["c3"] == 128
    assert cfg["model"]["dims"]["deconv_channels"] == 64
    assert cfg["model"]["decoder"]["num_blocks_s2"] == 4
    assert cfg["model"]["decoder"]["num_blocks_s1"] == 6
    assert cfg["loss"]["lambda_char"] == 0.8
    assert cfg["loss"]["lambda_fft"] == 0.05


def test_schema_rejects_unknown_registry_or_invalid_constructor_args() -> None:
    """Verifies that `validate_config` catches invalid registry names and invalid constructor kwargs."""
    cfg = load_config(PROJECT_ROOT / "configs/ablations/E_cwdnet_full.yaml")

    bad_reg = dict(cfg)
    bad_reg["model"] = dict(cfg["model"])
    bad_reg["model"]["fusion"] = {"name": "non_existent_fusion"}
    with pytest.raises(KeyError):
        validate_config(bad_reg)

    bad_param = load_config(PROJECT_ROOT / "configs/ablations/E_cwdnet_full.yaml")
    bad_param["model"]["prior_extractor"]["frf_temperature"] = 25.0
    with pytest.raises(TypeError):
        validate_config(bad_param)


def test_ast_registry_names_and_constructor_signatures_match_yamls() -> None:
    """
    Statically verifies via AST parsing (without needing `torch` installed) that:
      1. Every block registered in `src/models/` matches the exact keys in `config_schema.py` and YAMLs.
      2. Every parameter allowed in `config_schema.py` and passed in `configs/ablations/*.yaml`
         is an actual parameter of the corresponding class `__init__` method.
    """
    files = [
        PROJECT_ROOT / "src/models/classical/prior_extractor.py",
        PROJECT_ROOT / "src/models/blocks/encoder.py",
        PROJECT_ROOT / "src/models/blocks/daab.py",
        PROJECT_ROOT / "src/models/blocks/mtcsa.py",
        PROJECT_ROOT / "src/models/blocks/wiener_deconv.py",
        PROJECT_ROOT / "src/models/blocks/twin_decoder.py",
    ]
    all_registered: Dict[str, Tuple[str, Set[str]]] = {}
    for f in files:
        all_registered.update(_extract_registered_classes_and_init_args(f))

    # Check every ablation YAML against the actual AST __init__ signatures
    for cfg_rel in ABLATION_CONFIGS:
        cfg = load_config(PROJECT_ROOT / cfg_rel, validate=True)
        m = cfg["model"]
        for section in [
            "prior_extractor",
            "encoder",
            "alignment",
            "fusion",
            "deconv",
            "decoder",
        ]:
            sub = dict(m[section])
            name = sub.pop("name")
            assert name in all_registered, (
                f"{cfg_rel}: '{name}' in model.{section} is not registered in any class!"
            )
            cls_name, init_args = all_registered[name]
            unknown = set(sub.keys()) - init_args
            assert not unknown, (
                f"{cfg_rel}: model.{section} passes {unknown} not accepted by {cls_name}.__init__({init_args})"
            )

    # Check schema whitelists match AST __init__ signatures
    for name, allowed in VALID_PRIOR_PARAMS.items():
        assert allowed == all_registered[name][1]
    assert VALID_ENCODER_PARAMS == all_registered["shallow_multiscale"][1]
    for name, allowed in VALID_ALIGNMENT_PARAMS.items():
        assert allowed == all_registered[name][1]
    for name, allowed in VALID_FUSION_PARAMS.items():
        assert allowed == all_registered[name][1]
    assert VALID_DECONV_PARAMS == all_registered["wiener_fft_multichannel"][1]
    assert VALID_DECODER_PARAMS == all_registered["twin_naf_decoder"][1]


def test_composite_loss_targets_and_tv_field_via_ast() -> None:
    """
    Verifies statically on `src/losses/composite_loss.py` that:
      1. `CWDCompositeLoss.__init__` accepts all `VALID_LOSS_PARAMS`.
      2. `l_tilt` compares `flow_warp(tilt_only_flat, inv_tilt_flat)` against `j_gt` (not `tilt_only`).
      3. `l_tv` is evaluated on `output.inverse_tilt_field` (not `j_hat`).
    """
    loss_path = PROJECT_ROOT / "src/losses/composite_loss.py"
    src_text = loss_path.read_text(encoding="utf-8")
    tree = ast.parse(src_text)

    init_args: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "CWDCompositeLoss":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    init_args = {a.arg for a in item.args.args if a.arg != "self"}

    assert init_args == VALID_LOSS_PARAMS
    assert "flow_warp(tilt_only_flat, inv_tilt_flat)" in src_text
    assert "self.charbonnier(rectified_tilt, j_gt)" in src_text
    assert "self.tv_loss(output.inverse_tilt_field)" in src_text
    assert "self.tv_loss(j_hat)" not in src_text


@pytest.mark.parametrize("cfg_rel_path", ABLATION_CONFIGS)
def test_liskov_substitution_all_ablation_variants(cfg_rel_path: str) -> None:
    """
    Verifies that every ablation variant (A through E) can be instantiated via `build_cwd_net`
    and produces identical output contract shapes (`ModelOutput`).
    """
    torch = pytest.importorskip("torch")
    from src.models.cwd_net import build_cwd_net

    cfg = load_config(PROJECT_ROOT / cfg_rel_path)
    cfg["model"]["dims"] = {"c1": 16, "c2": 24, "c3": 32, "deconv_channels": 16}
    cfg["model"]["decoder"]["num_blocks_s2"] = 1
    cfg["model"]["decoder"]["num_blocks_s1"] = 1
    model = build_cwd_net(cfg).eval()

    b, t, c, h, w = 1, 4, 3, 64, 64
    x = torch.rand(b, t, c, h, w)
    with torch.no_grad():
        out = model(x)

    assert out.restored.shape == (b, t, c, h, w)
    assert out.rectified_rgb.shape == (b, t, c, h, w)
    assert out.inverse_tilt_field.shape == (b, t, 2, h, w)
    assert out.frf_reference.shape == (b, c, h, w)
    assert torch.isfinite(out.restored).all()
