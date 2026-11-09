#!/usr/bin/env python3
"""Main CLI Training Entrypoint for CWD-Net and Ablation Variants A-E."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.datasets import ATSynDataset, collate_video_batch
from src.engine.trainer import Trainer
from src.models.cwd_net import build_cwd_net
from src.utils.config_loader import load_config, save_config
from src.utils.reproducibility import (
    collect_env_and_git_info,
    seed_everything,
    seed_worker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CWD-Net (CLEAR-Wiener-DATUM Network) for Atmospheric Turbulence Mitigation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ablations/E_cwdnet_full.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional explicit path to checkpoint (.pt) to resume training from.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Automatically resume from <exp-dir>/checkpoints/last.pt if it exists (ideal for Google Drive).",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Optional explicit experiment output directory (e.g. on Google Drive).",
    )
    parser.add_argument(
        "--opts",
        nargs="*",
        default=[],
        help="Optional dot-list config overrides, e.g., train.total_iters=1000 experiment.use_wandb=true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.opts)

    exp_cfg = cfg.get("experiment", {})
    seed = int(exp_cfg.get("seed", 42))
    deterministic = bool(exp_cfg.get("deterministic", False))
    seed_everything(seed=seed, deterministic=deterministic)

    if args.exp_dir is not None:
        exp_dir = Path(args.exp_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = exp_cfg.get("name", "cwdnet")
        output_root = Path(exp_cfg.get("output_root", "experiments"))
        exp_dir = output_root / f"{timestamp}_{exp_name}_seed{seed}"

    exp_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, exp_dir / "config_resolved.yaml")
    env_info = collect_env_and_git_info(exp_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CWD-Net] Experiment directory: {exp_dir}")
    print(f"[CWD-Net] Device: {device} ({env_info.get('device_name')})")

    # Build Datasets & DataLoaders
    data_cfg = cfg.get("data", {})
    train_dataset = ATSynDataset(
        root_dir=data_cfg.get("train_root", "datasets/ATSyn_dynamic/train"),
        num_frames=int(data_cfg.get("num_frames", 12)),
        crop_size=int(data_cfg.get("crop_size", 256)),
        is_train=True,
        allow_synthetic_fallback=bool(data_cfg.get("allow_synthetic_fallback", True)),
        fallback_length=int(data_cfg.get("fallback_length", 64)),
    )
    val_dataset = ATSynDataset(
        root_dir=data_cfg.get("val_root", "datasets/ATSyn_dynamic/val"),
        num_frames=int(data_cfg.get("num_frames", 12)),
        crop_size=int(data_cfg.get("crop_size", 256)),
        is_train=False,
        allow_synthetic_fallback=bool(data_cfg.get("allow_synthetic_fallback", True)),
        fallback_length=max(4, int(data_cfg.get("fallback_length", 64)) // 8),
    )

    g = torch.Generator()
    g.manual_seed(seed)

    batch_size = int(data_cfg.get("batch_size", 2))
    num_workers = int(data_cfg.get("num_workers", 4))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_video_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_video_batch,
    )

    # Build Model
    model = build_cwd_net(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[CWD-Net] Trainable parameters: {total_params / 1e6:.3f} M")

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        exp_dir=exp_dir,
        device=device,
    )

    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        candidate = exp_dir / "checkpoints" / "last.pt"
        if candidate.exists():
            resume_path = str(candidate)

    if resume_path is not None:
        print(f"[CWD-Net] Resuming from checkpoint: {resume_path}")
        trainer.resume(resume_path)

    final_metrics = trainer.fit()
    print(f"[CWD-Net] Training finished. Final validation metrics: {final_metrics}")


if __name__ == "__main__":
    main()
