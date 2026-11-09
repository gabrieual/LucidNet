#!/usr/bin/env python3
"""Production Video Inference CLI Script supporting .mp4 files & frame folders."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision.utils import save_image

from src.data.datasets import (
    VIDEO_EXTENSIONS,
    _list_image_files,
    _list_video_files,
    _load_all_video_frames,
    _load_rgb_tensor,
)
from src.engine.inference import VideoInferenceEngine
from src.models.cwd_net import build_cwd_net
from src.utils.config_loader import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CWD-Net inference on an .mp4 video or folder of frames (Full HD supported via Hann Overlap-Tile)."
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
        help="Path to a degraded .mp4 video file OR directory containing .mp4/image frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write restored frames and .mp4 video.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "forensic_snapshot"],
        default="video",
        help="'video' saves all restored frames + .mp4; 'forensic_snapshot' saves single sharpest central frame.",
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


def _save_mp4_if_possible(frames: torch.Tensor, mp4_path: Path, fps: float = 30.0) -> None:
    """Writes [T, 3, H, W] float32 in [0, 1] to an .mp4 file using OpenCV when available."""
    try:
        import cv2  # type: ignore
        import numpy as np

        t, _, h, w = frames.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (w, h))
        for idx in range(t):
            rgb = (
                (frames[idx].permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0)
                .round()
                .astype(np.uint8)
            )
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()
        print(f"[Infer] Saved restored MP4 video to {mp4_path}")
    except ImportError:
        pass


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

    in_path = Path(args.input_dir)
    if in_path.is_file() and in_path.suffix.lower() in VIDEO_EXTENSIONS:
        video_tensor = _load_all_video_frames(in_path)
    elif in_path.is_dir():
        vid_files = _list_video_files(in_path)
        if len(vid_files) > 0:
            video_tensor = _load_all_video_frames(vid_files[0])
        else:
            frame_paths = _list_image_files(in_path)
            if len(frame_paths) == 0:
                raise RuntimeError(f"No .mp4 videos or image frames found in {in_path}")
            video_tensor = torch.stack([_load_rgb_tensor(p) for p in frame_paths], dim=0)
    else:
        raise RuntimeError(f"Input path does not exist: {in_path}")

    print(
        f"[Infer] Loaded {video_tensor.shape[0]} frames of shape {tuple(video_tensor.shape[1:])}"
    )

    inf_cfg = cfg.get("inference", {})
    engine = VideoInferenceEngine(
        model=model,
        device=device,
        window_size=int(inf_cfg.get("window_size", 24)),
        temporal_stride=int(inf_cfg.get("temporal_stride", 16)),
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
        _save_mp4_if_possible(restored, out_dir / "restored.mp4")
        print(f"[Infer] Saved {restored.shape[0]} restored frames to {out_dir}")


if __name__ == "__main__":
    main()
