"""Utilities package for CWD-Net with lazy loading for heavy dependencies."""

from typing import Any
from src.utils.config_loader import deep_merge, load_config, save_config
from src.utils.config_schema import validate_config

__all__ = [
    "deep_merge",
    "load_config",
    "save_config",
    "validate_config",
    "seed_everything",
    "seed_worker",
    "collect_env_and_git_info",
    "save_comparison_grid",
    "save_psf_grid",
]


def __getattr__(name: str) -> Any:
    if name in {"seed_everything", "seed_worker", "collect_env_and_git_info"}:
        from src.utils import reproducibility

        return getattr(reproducibility, name)
    if name in {"save_comparison_grid", "save_psf_grid"}:
        from src.utils import visualization

        return getattr(visualization, name)
    raise AttributeError(f"module 'src.utils' has no attribute {name!r}")
