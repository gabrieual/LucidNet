#!/usr/bin/env python3
"""Production Video Inference CLI Script supporting Sliding Window, Hann Tiling & Forensic Mode."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision.utils import save_image

from src.data.datasets import _list_image_files, _load_rgb_tensor
from src.engine.inference import VideoInferenceEngine
from src.models.cwd_net import build_cwd_net
from src.utils.config_loader import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CWD-Net inference on a folder of frames (Full HD supported via Hann Overlap-Tile)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ablations/E_cwdnet_full.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint (.pt). Loads EMA weights automatically.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing degraded video frames (.png/.jpg).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write restored frames.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "forensic_snapshot"],
        default="video",
        help="'video' saves all restored frames; 'forensic_snapshot' saves single sharpest central frame.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Spatial tile size for 2D Hann Overlap-Tile inference.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=64,
        help="Spatial overlap in pixels between adjacent tiles.",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_cwd_net(cfg).to(device).eval()
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state_dict)
        print(f"[Infer] Loaded EMA weights from {args.checkpoint}")

    in_dir = Path(args.input_dir)
    frame_paths = _list_image_files(in_dir)
    if len(frame_paths) == 0:
        raise RuntimeError(f"No image frames found in {in_dir}")

    video_tensor = torch.stack([_load_rgb_tensor(p) for p in frame_paths], dim=0)  # [T, 3, H, W]
    print(f"[Infer] Loaded {video_tensor.shape[0]} frames of shape {tuple(video_tensor.shape[1:])}")

    inf_cfg = cfg.get("inference", {})
    engine = VideoInferenceEngine(
        model=model,
        device=device,
        window_size=int(inf_cfg.get("window_size", 12)),
        temporal_stride=int(inf_cfg.get("temporal_stride", 6)),
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        use_amp=True,
    )

    restored = engine.infer_video(video_tensor, mode=args.mode)  # [T_out, 3, H, W]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "forensic_snapshot":
        out_file = out_dir / "forensic_snapshot.png"
        save_image(restored[0], str(out_file))
        print(f"[Infer] Saved forensic snapshot to {out_file}")
    else:
        for idx, frame in enumerate(restored):
            out_file = out_dir / f"frame_{idx:05d}.png"
            save_image(frame, str(out_file))
        print(f"[Infer] Saved {restored.shape[0]} restored frames to {out_dir}")


if __name__ == "__main__":
    main()
