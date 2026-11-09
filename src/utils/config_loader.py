"""YAML Configuration Loader supporting hierarchical `_base_` inheritance, CLI overrides & Schema Validation."""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from src.utils.config_schema import validate_config


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges `override` dictionary into a copy of `base`.
    If both `base[k]` and `override[k]` are component config dicts with a `"name"` key
    and the component `"name"` changes (e.g., swapping `"daab_grum_dim"` for `"optical_flow_only"`
    in an ablation), replaces the component dict cleanly so hyperparameters from the old
    component do not leak into the new component's constructor.
    """
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            if "name" in result[k] and "name" in v and result[k]["name"] != v["name"]:
                result[k] = copy.deepcopy(v)
            else:
                result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _load_raw_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    if "_base_" in cfg:
        base_rel = cfg.pop("_base_")
        base_path = (path.parent / base_rel).resolve()
        base_cfg = _load_raw_config(base_path)
        cfg = deep_merge(base_cfg, cfg)
    return cfg


def load_config(
    config_path: str | Path,
    overrides: Optional[List[str]] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Loads a YAML configuration file, recursively resolving `_base_` parent files,
    applying optional dot-separated CLI key=value overrides, and validating against
    the strict schema & constructor signatures in `src.utils.config_schema`.
    """
    cfg = _load_raw_config(config_path)

    if overrides:
        for item in overrides:
            if "=" not in item:
                continue
            key_path, raw_val = item.split("=", 1)
            parsed_val = yaml.safe_load(raw_val)
            keys = key_path.strip().split(".")
            curr = cfg
            for k in keys[:-1]:
                if k not in curr or not isinstance(curr[k], dict):
                    curr[k] = {}
                curr = curr[k]
            last_k = keys[-1]
            if last_k == "name" and "name" in curr and curr["name"] != parsed_val:
                curr.clear()
            curr[last_k] = parsed_val

    if validate and "model" in cfg:
        validate_config(cfg)

    return cfg


def save_config(cfg: Dict[str, Any], save_path: str | Path) -> None:
    """Saves the resolved configuration dictionary as a YAML file."""
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
