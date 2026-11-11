"""Atomic Checkpoint Management with EMA, Full RNG Recovery & Weights & Biases Cloud Sync."""

import copy
import os
import random
import shutil
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
    typically yields +0.15 to +0.35 dB PSNR and superior temporal stability.
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
    Manages atomic checkpoint saving, exact training state restoration, and
    automatic Weights & Biases (`wandb`) cloud synchronization:
      - Writes to `<filename>.tmp` and calls `os.replace()` (POSIX atomic rename)
        so power loss or SIGKILL never corrupts an existing checkpoint.
      - Tracks `last.pt`, `best_psnr.pt`, `best_cw_ssim.pt`, and rolling `step_{N}.pt`.
      - When `wandb_run` is active:
          1. Uploads `last.pt`, `best_psnr.pt`, and `best_cw_ssim.pt` to W&B run files
             (`policy="now"`) so `wandb.restore()` can recover them from any machine.
          2. Logs versioned W&B Model Artifacts (`type="model"`) with aliases
             `["latest", "best_psnr", "best_cw_ssim"]`.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        keep_last_k: int = 3,
        wandb_run: Optional[Any] = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = keep_last_k
        self.wandb_run = wandb_run
        self.best_psnr: float = -float("inf")
        self.best_cw_ssim: float = -float("inf")
        self._rolling_History: List[Path] = []

    @staticmethod
    def _atomic_save(payload: Dict[str, Any], target_path: Path) -> None:
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, target_path)

    def _sync_to_wandb(
        self,
        files_to_sync: List[Path],
        aliases: List[str],
        global_step: int,
        metrics: Dict[str, float],
    ) -> None:
        """Uploads checkpoint files and model artifacts to Weights & Biases."""
        if self.wandb_run is None:
            return
        try:
            import wandb  # type: ignore

            base_dir = str(self.checkpoint_dir.parent)
            for fpath in files_to_sync:
                if fpath.exists():
                    self.wandb_run.save(
                        str(fpath),
                        base_path=base_dir,
                        policy="now",
                    )

            # Log versioned W&B Artifact whenever a new best checkpoint is achieved
            best_Aliases = [a for a in aliases if a.startswith("best_")]
            if best_Aliases:
                art_name = f"cwdnet-{self.wandb_run.id}"
                artifact = wandb.Artifact(
                    name=art_name,
                    type="model",
                    metadata={
                        "global_step": global_step,
                        "best_psnr": self.best_psnr,
                        "best_cw_ssim": self.best_cw_ssim,
                        **metrics,
                    },
                )
                for a in best_Aliases:
                    pt_file = self.checkpoint_dir / f"{a}.pt"
                    if pt_file.exists():
                        artifact.add_file(str(pt_file), name=f"{a}.pt")
                self.wandb_run.log_artifact(artifact, aliases=["latest"] + best_Aliases)
        except Exception as exc:
            print(f"[CheckpointManager] Warning: W&B checkpoint sync failed ({exc}).")

    def try_restore_from_wandb(self, filename: str = "last.pt") -> Optional[Path]:
        """
        Attempts to download `checkpoints/<filename>` from the active W&B run
        (or its latest model artifact) if it is not present on local disk.
        """
        target_path = self.checkpoint_dir / filename
        if target_path.exists():
            return target_path
        if self.wandb_run is None:
            return None

        try:
            import wandb  # type: ignore

            rel_name = f"checkpoints/{filename}"
            restored_file = wandb.restore(
                rel_name,
                run_path=self.wandb_run.path,
                root=str(self.checkpoint_dir.parent),
                replace=True,
            )
            if restored_file is not None and hasattr(restored_file, "name"):
                restored_path = Path(restored_file.name)
                restored_file.close()
                if restored_path.exists():
                    if restored_path.resolve() != target_path.resolve():
                        shutil.copy2(restored_path, target_path)
                    print(
                        f"[CheckpointManager] Restored '{filename}' from W&B cloud ({target_path})"
                    )
                    return target_path
        except Exception:
            pass

        # Fallback: try downloading from W&B Model Artifact
        try:
            art_name = f"cwdnet-{self.wandb_run.id}:latest"
            artifact = self.wandb_run.use_artifact(art_name, type="model")
            art_dir = Path(artifact.download())
            candidate = art_dir / filename
            if candidate.exists():
                shutil.copy2(candidate, target_path)
                print(
                    f"[CheckpointManager] Restored '{filename}' from W&B Artifact '{art_name}'"
                )
                return target_path
        except Exception:
            pass

        return None

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
            "ema_state_dict": (
                ema.state_dict() if ema is not None else raw_model.state_dict()
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
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
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
        }

        last_path = self.checkpoint_dir / "last.pt"
        self._atomic_save(payload, last_path)
        files_to_sync = [last_path]
        aliases = ["latest"]

        if is_best_psnr:
            best_psnr_path = self.checkpoint_dir / "best_psnr.pt"
            self._atomic_save(payload, best_psnr_path)
            files_to_sync.append(best_psnr_path)
            aliases.append("best_psnr")

        if is_best_cw_ssim:
            best_cw_path = self.checkpoint_dir / "best_cw_ssim.pt"
            self._atomic_save(payload, best_cw_path)
            files_to_sync.append(best_cw_path)
            aliases.append("best_cw_ssim")

        if is_periodic_step:
            step_path = self.checkpoint_dir / f"step_{global_step:07d}.pt"
            self._atomic_save(payload, step_path)
            self._rolling_History.append(step_path)
            while len(self._rolling_History) > self.keep_last_k:
                old = self._rolling_History.pop(0)
                if old.exists():
                    old.unlink()

        # Upload checkpoints to Weights & Biases cloud
        self._sync_to_wandb(
            files_to_sync=files_to_sync,
            aliases=aliases,
            global_step=global_step,
            metrics=metrics,
        )

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
