"""Atomic Checkpoint Management with Exponential Moving Average (EMA) & Full RNG Recovery."""

import copy
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn


class ModelEMA:
    """
    Maintains an Exponential Moving Average (EMA) of model parameters:
        theta_ema <- decay * theta_ema + (1 - decay) * theta
    In optical restoration and blind deconvolution, evaluating with EMA weights
    typically yields +0.25 to +0.40 dB PSNR and superior temporal stability.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, ema_v in self.ema_model.state_dict().items():
            model_v = msd[k].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
            else:
                ema_v.copy_(model_v)

    def state_dict(self) -> Dict[str, Any]:
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.ema_model.load_state_dict(state_dict)


class CheckpointManager:
    """
    Manages atomic checkpoint saving and exact training state restoration
    as specified in Section 7.1 of `arquitetura.md`:
      - Writes to `<filename>.tmp` and calls `os.replace()` (POSIX atomic rename)
        so power loss or SIGKILL never corrupts an existing checkpoint.
      - Tracks `last.pt`, `best_psnr.pt`, `best_cw_ssim.pt`, and rolling `step_{N}.pt`.
    """

    def __init__(self, checkpoint_dir: str | Path, keep_last_k: int = 3) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = keep_last_k
        self.best_psnr: float = -float("inf")
        self.best_cw_ssim: float = -float("inf")
        self._rolling_History: List[Path] = []

    @staticmethod
    def _atomic_save(payload: Dict[str, Any], target_path: Path) -> None:
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, target_path)

    def save(
        self,
        epoch: int,
        global_step: int,
        model: nn.Module,
        ema: Optional[ModelEMA],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        scaler: Optional[torch.amp.GradScaler],
        config: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        is_periodic_step: bool = False,
    ) -> None:
        metrics = metrics or {}
        psnr = metrics.get("val/psnr", self.best_psnr)
        cw_ssim = metrics.get("val/cw_ssim", self.best_cw_ssim)

        is_best_psnr = psnr > self.best_psnr
        is_best_cw_ssim = cw_ssim > self.best_cw_ssim
        if is_best_psnr:
            self.best_psnr = psnr
        if is_best_cw_ssim:
            self.best_cw_ssim = cw_ssim

        raw_model = model.module if hasattr(model, "module") else model
        payload: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": raw_model.state_dict(),
            "ema_state_dict": ema.state_dict() if ema is not None else raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_metrics": {
                "psnr": self.best_psnr,
                "cw_ssim": self.best_cw_ssim,
            },
            "config": config,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
            },
        }

        # 1. Always update `last.pt` atomically
        self._atomic_save(payload, self.checkpoint_dir / "last.pt")

        # 2. Update best checkpoints when improved
        if is_best_psnr:
            self._atomic_save(payload, self.checkpoint_dir / "best_psnr.pt")
        if is_best_cw_ssim:
            self._atomic_save(payload, self.checkpoint_dir / "best_cw_ssim.pt")

        # 3. Rolling step checkpoints (keep last K)
        if is_periodic_step:
            step_path = self.checkpoint_dir / f"step_{global_step:07d}.pt"
            self._atomic_save(payload, step_path)
            self._rolling_History.append(step_path)
            while len(self._rolling_History) > self.keep_last_k:
                old = self._rolling_History.pop(0)
                if old.exists():
                    old.unlink()

    def load(
        self,
        checkpoint_path: str | Path,
        model: nn.Module,
        ema: Optional[ModelEMA] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        restore_rng: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> Dict[str, Any]:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(ckpt["model_state_dict"])

        if ema is not None and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        best_m = ckpt.get("best_metrics", {})
        self.best_psnr = best_m.get("psnr", self.best_psnr)
        self.best_cw_ssim = best_m.get("cw_ssim", self.best_cw_ssim)

        if restore_rng and "rng_state" in ckpt:
            rng = ckpt["rng_state"]
            if rng.get("python") is not None:
                random.setstate(rng["python"])
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"].cpu())
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                try:
                    torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
                except Exception:
                    pass

        return ckpt
