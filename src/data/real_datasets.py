"""Loaders for Real Atmospheric Turbulence Video Benchmarks (.mp4 videos & frame folders)."""

from pathlib import Path
from typing import List, Optional
import torch
from torch.utils.data import Dataset

from src.contracts.batch import VideoBatch
from src.data.atsyn_dataset import (
    VIDEO_EXTENSIONS,
    _list_image_files,
    _list_video_files,
    _load_all_video_frames,
    _load_rgb_tensor,
)


class RealTurbulenceVideoDataset(Dataset):
    """
    Inference / Blind Evaluation dataset for real atmospheric turbulence sequences
    (e.g., OTIS, BRIAR, RLR-AT, TurbText, CLEAR benchmark).
    Supports:
      - A folder of `.mp4` videos (`root_dir/<clip_id>.mp4`)
      - A single `.mp4` video file (`root_dir = path/to/video.mp4`)
      - A folder of clip subdirectories (`root_dir/<clip_id>/<frame.png>`)
      - A flat folder of image frames (`root_dir/<frame.png>`).
    """

    def __init__(self, root_dir: str, max_frames: Optional[int] = None) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.max_frames = max_frames
        self.clip_entries: List[Path] = []

        if self.root_dir.exists():
            if self.root_dir.is_file() and self.root_dir.suffix.lower() in VIDEO_EXTENSIONS:
                self.clip_entries = [self.root_dir]
            elif self.root_dir.is_dir():
                videos = _list_video_files(self.root_dir)
                if len(videos) > 0:
                    self.clip_entries = videos
                else:
                    subdirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
                    if len(subdirs) > 0 and len(_list_image_files(subdirs[0])) > 0:
                        self.clip_entries = subdirs
                    elif len(_list_image_files(self.root_dir)) > 0:
                        self.clip_entries = [self.root_dir]

    def __len__(self) -> int:
        return len(self.clip_entries)

    def __getitem__(self, idx: int) -> VideoBatch:
        entry = self.clip_entries[idx]
        if entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS:
            frames = _load_all_video_frames(entry, max_frames=self.max_frames)
            clip_name = entry.stem
        else:
            files = _list_image_files(entry)
            if self.max_frames is not None:
                files = files[: self.max_frames]
            frames = torch.stack([_load_rgb_tensor(f) for f in files], dim=0)
            clip_name = entry.name
        return VideoBatch(
            degraded=frames,
            gt=None,
            tilt_only=None,
            meta={"clip_name": clip_name},
        )
