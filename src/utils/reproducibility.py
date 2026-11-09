"""Reproducibility, Deterministic Seeding, and Git/Environment Provenance Utilities."""

import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """
    Seeds Python, NumPy, and PyTorch (CPU & CUDA) RNGs and configures CuDNN/TF32 flags
    as specified in Section 6.2 of `arquitetura.md`.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn ensuring unique, reproducible seeds across multi-process workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def collect_env_and_git_info(output_dir: Path) -> Dict[str, Any]:
    """
    Records git commit hash, git diff (uncommitted changes), Python/PyTorch/CUDA versions,
    and GPU hardware metadata into `output_dir` for strict scientific provenance.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        ),
    }

    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        info["git_commit"] = commit
        diff = subprocess.check_output(
            ["git", "diff"], stderr=subprocess.DEVNULL
        ).decode("utf-8")
        (output_dir / "git_diff.patch").write_text(diff, encoding="utf-8")
    except Exception:
        info["git_commit"] = "not_a_git_repo"

    env_lines = [f"{k}: {v}" for k, v in info.items()]
    (output_dir / "env_info.txt").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    return info
