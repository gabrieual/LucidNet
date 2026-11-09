"""Loaders for Real Atmospheric Turbulence Video Benchmarks (OTIS, BRIAR, RLR-AT, TurbText)."""

from pathlib import Path
from typing import List, Optional
import torch
from torch.utils.data import Dataset

from src.contracts.batch import VideoBatch
from src.data.atsyn_dataset import _list_image_files, _load_rgb_tensor


class RealTurbulenceVideoDataset(Dataset):
    """
    Inference / Blind Evaluation dataset for real atmospheric turbulence sequences
    (e.g., OTIS, BRIAR, RLR-AT, TurbText, CLEAR benchmark).
    Supports a folder of clips (`root_dir/<clip_id>/<frame.png>`) or a flat folder of frames.
    """

    def __init__(self, root_dir: str, max_frames: Optional[int] = None) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.max_frames = max_frames
        self.clip_dirs: List[Path] = []

        if self.root_dir.exists():
            subdirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
            if len(subdirs) > 0 and len(_list_image_files(subdirs[0])) > 0:
                self.clip_dirs = subdirs
            elif len(_list_image_files(self.root_dir)) > 0:
                self.clip_dirs = [self.root_dir]

    def __len__(self) -> int:
        return len(self.clip_dirs)

    def __getitem__(self, idx: int) -> VideoBatch:
        clip_dir = self.clip_dirs[idx]
        files = _list_image_files(clip_dir)
        if self.max_frames is not None:
            files = files[: self.max_frames]
        frames = torch.stack([_load_rgb_tensor(f) for f in files], dim=0)
        return VideoBatch(
            degraded=frames,
            gt=None,
            tilt_only=None,
            meta={"clip_name": clip_dir.name},
        )
