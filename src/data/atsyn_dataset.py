"""Loader for ATSyn-Static and ATSyn-Dynamic Paired Benchmarks."""

from pathlib import Path
from typing import List, Optional, Tuple
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

from src.contracts.batch import VideoBatch
from src.data.p2s_simulator import PhysicsTurbulenceSimulator

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_image_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return TF.to_tensor(rgb)  # [3, H, W] float32 in [0, 1]


def collate_video_batch(samples: List[VideoBatch]) -> VideoBatch:
    """Custom collate_fn assembling a list of per-clip VideoBatch objects into a batched VideoBatch."""
    degraded = torch.stack([s.degraded for s in samples], dim=0)
    gt = (
        torch.stack([s.gt for s in samples if s.gt is not None], dim=0)
        if samples[0].gt is not None
        else None
    )
    tilt_only = (
        torch.stack([s.tilt_only for s in samples if s.tilt_only is not None], dim=0)
        if samples[0].tilt_only is not None
        else None
    )
    meta = {
        "clip_names": [
            s.meta.get("clip_name", f"clip_{i}") for i, s in enumerate(samples)
        ],
        "d_r0": [s.meta.get("d_r0", 0.0) for s in samples],
    }
    return VideoBatch(degraded=degraded, gt=gt, tilt_only=tilt_only, meta=meta)


class ATSynDataset(Dataset):
    """
    Paired atmospheric turbulence video dataset (ATSyn-static, ATSyn-dynamic, HeatChamber, TUG).
    Expected directory layout:
        root_dir/
          degraded/<clip_id>/<frame_000.png>
          gt/<clip_id>/<frame_000.png>
          tilt_only/<clip_id>/<frame_000.png>  (optional intermediate tilt-only frames I_tilt)
    If `root_dir` does not exist or is empty and `allow_synthetic_fallback=True`,
    automatically falls back to procedural synthetic P2S generation so scripts run out-of-the-box.
    """

    def __init__(
        self,
        root_dir: str,
        num_frames: int = 12,
        crop_size: int = 256,
        is_train: bool = True,
        allow_synthetic_fallback: bool = True,
        fallback_length: int = 64,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.is_train = is_train
        self.allow_synthetic_fallback = allow_synthetic_fallback
        self.fallback_length = fallback_length

        self.clips: List[Tuple[Path, Path, Optional[Path]]] = []
        deg_root = self.root_dir / "degraded"
        gt_root = self.root_dir / "gt"
        tilt_root = self.root_dir / "tilt_only"

        if deg_root.exists() and gt_root.exists():
            for clip_dir in sorted(deg_root.iterdir()):
                if not clip_dir.is_dir():
                    continue
                gt_clip = gt_root / clip_dir.name
                if gt_clip.exists():
                    tilt_clip = (
                        (tilt_root / clip_dir.name)
                        if (tilt_root / clip_dir.name).exists()
                        else None
                    )
                    self.clips.append((clip_dir, gt_clip, tilt_clip))

        self.simulator = PhysicsTurbulenceSimulator()

    def __len__(self) -> int:
        if len(self.clips) > 0:
            return len(self.clips)
        if self.allow_synthetic_fallback:
            return self.fallback_length
        return 0

    def _generate_procedural_clean_clip(self, idx: int) -> torch.Tensor:
        """Generates a crisp multi-frequency Siemens/checkerboard/edge target clip for fallback."""
        h = w = self.crop_size
        t = self.num_frames
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h),
            torch.linspace(-1.0, 1.0, w),
            indexing="ij",
        )
        r = torch.sqrt(xx.square() + yy.square() + 1e-6)
        theta = torch.atan2(yy, xx)
        freq = 6.0 + (idx % 7) * 2.0
        star = torch.sigmoid(8.0 * torch.cos(freq * theta) * torch.exp(-r * 0.8))
        bars = ((torch.sin(xx * 25.0) * torch.cos(yy * 25.0)) > 0).float() * 0.6
        base = torch.stack(
            [
                0.6 * star + 0.4 * bars,
                0.5 * star + 0.3 * (1.0 - r.clamp(0, 1)),
                0.7 * bars + 0.3 * star,
            ],
            dim=0,
        ).clamp(0.0, 1.0)
        return base.unsqueeze(0).repeat(t, 1, 1, 1)  # [T, 3, H, W]

    def __getitem__(self, idx: int) -> VideoBatch:
        if len(self.clips) == 0:
            clean_clip = self._generate_procedural_clean_clip(idx)
            degraded, tilt_only, d_r0 = self.simulator(clean_clip)
            return VideoBatch(
                degraded=degraded,
                gt=clean_clip,
                tilt_only=tilt_only,
                meta={"clip_name": f"synthetic_p2s_{idx:04d}", "d_r0": d_r0},
            )

        deg_dir, gt_dir, tilt_dir = self.clips[idx]
        deg_files = _list_image_files(deg_dir)
        gt_files = _list_image_files(gt_dir)
        tilt_files = _list_image_files(tilt_dir) if tilt_dir is not None else []

        total = min(len(deg_files), len(gt_files))
        if total >= self.num_frames:
            start = (
                int(torch.randint(0, total - self.num_frames + 1, (1,)).item())
                if self.is_train
                else 0
            )
            indices = list(range(start, start + self.num_frames))
        else:
            indices = [i % total for i in range(self.num_frames)]

        deg_seq = torch.stack([_load_rgb_tensor(deg_files[i]) for i in indices], dim=0)
        gt_seq = torch.stack([_load_rgb_tensor(gt_files[i]) for i in indices], dim=0)
        tilt_seq = (
            torch.stack([_load_rgb_tensor(tilt_files[i]) for i in indices], dim=0)
            if len(tilt_files) >= total
            else None
        )

        # Synchronized spatial crop & random flips during training
        _, _, h, w = deg_seq.shape
        cs = self.crop_size
        if cs > 0 and h >= cs and w >= cs:
            if self.is_train:
                top = int(torch.randint(0, h - cs + 1, (1,)).item())
                left = int(torch.randint(0, w - cs + 1, (1,)).item())
            else:
                top = (h - cs) // 2
                left = (w - cs) // 2
            deg_seq = deg_seq[:, :, top : top + cs, left : left + cs]
            gt_seq = gt_seq[:, :, top : top + cs, left : left + cs]
            if tilt_seq is not None:
                tilt_seq = tilt_seq[:, :, top : top + cs, left : left + cs]

        if self.is_train and torch.rand(1).item() > 0.5:
            deg_seq = torch.flip(deg_seq, dims=[-1])
            gt_seq = torch.flip(gt_seq, dims=[-1])
            if tilt_seq is not None:
                tilt_seq = torch.flip(tilt_seq, dims=[-1])

        return VideoBatch(
            degraded=deg_seq,
            gt=gt_seq,
            tilt_only=tilt_seq,
            meta={"clip_name": deg_dir.name, "d_r0": 0.0},
        )
