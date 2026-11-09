"""Loader for ATSyn-Static and ATSyn-Dynamic Paired Benchmarks (.mp4 videos & image folders)."""

from pathlib import Path
from typing import List, Optional, Tuple
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

from src.contracts.batch import VideoBatch
from src.data.p2s_simulator import PhysicsTurbulenceSimulator

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _list_image_files(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])


def _list_video_files(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS])


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return TF.to_tensor(rgb)  # [3, H, W] float32 in [0, 1]


def _get_video_frame_count(video_path: Path) -> int:
    """Returns total number of frames in an .mp4 video file without decoding all frames."""
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if count > 0:
            return count
    except ImportError:
        pass

    from torchvision.io import read_video

    frames, _, _ = read_video(str(video_path), pts_unit="sec", output_format="TCHW")
    return int(frames.shape[0])


def _load_video_clip_frames(
    video_path: Path,
    start_frame: int,
    num_frames: int,
) -> torch.Tensor:
    """
    Decodes `num_frames` consecutive RGB frames starting at `start_frame` from an `.mp4` file.
    Returns tensor of shape `[T, 3, H, W]` in `float32` `[0, 1]`.
    Uses fast OpenCV seeking (`cv2.VideoCapture`) when available so only the requested
    `num_frames` window is decoded; falls back to `torchvision.io.read_video`.
    """
    try:
        import cv2  # type: ignore
        import numpy as np

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0 and start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames_list: List[torch.Tensor] = []
        for _ in range(num_frames):
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy( np.ascontiguousarray(frame_rgb) ).permute(2, 0, 1).float() / 255.0
            frames_list.append(tensor)
        cap.release()

        if len(frames_list) > 0:
            while len(frames_list) < num_frames:
                frames_list.append(frames_list[-1].clone())
            return torch.stack(frames_list[:num_frames], dim=0)
    except ImportError:
        pass

    # Fallback to torchvision.io.read_video
    from torchvision.io import read_video

    vframes, _, _ = read_video(str(video_path), pts_unit="sec", output_format="TCHW")
    total_v = vframes.shape[0]
    if total_v == 0:
        raise RuntimeError(f"Could not decode any frames from video: {video_path}")
    indices = [(start_frame + i) % total_v for i in range(num_frames)]
    selected = vframes[indices].float() / 255.0
    return selected


def _load_all_video_frames(
    video_path: Path, max_frames: Optional[int] = None
) -> torch.Tensor:
    """Loads all frames (or up to `max_frames`) from an `.mp4` video file for inference."""
    total = _get_video_frame_count(video_path)
    n = min(total, max_frames) if (max_frames is not None and total > 0) else max(1, total)
    return _load_video_clip_frames(video_path, start_frame=0, num_frames=n)


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
    Paired atmospheric turbulence video dataset supporting BOTH:
      1. Direct `.mp4` video files inside `turb/`, `tilt/`, and `gt/` (official ATSyn layout):
           train/ (or test/)
             ├── turb/<clip_id>.mp4
             ├── tilt/<clip_id>.mp4
             └── gt/<clip_id>.mp4 (or .png/.jpg for static targets)
      2. Frame directories inside `turb/<clip_id>/*.png`, `tilt/<clip_id>/*.png`, `gt/<clip_id>/*.png`.

    Also automatically resolves `val` -> `test` when the dataset only has `train/` and `test/` splits.
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
        raw_root = Path(root_dir)
        # If `.../val` is requested but ATSyn only has `train` and `test`, automatically use `.../test`
        if not raw_root.exists() and raw_root.name == "val" and (raw_root.parent / "test").exists():
            raw_root = raw_root.parent / "test"
        self.root_dir = raw_root
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.is_train = is_train
        self.allow_synthetic_fallback = allow_synthetic_fallback
        self.fallback_length = fallback_length

        self.clips: List[Tuple[Path, Path, Optional[Path]]] = []
        deg_root = (
            (self.root_dir / "turb")
            if (self.root_dir / "turb").exists()
            else (self.root_dir / "degraded")
        )
        gt_root = self.root_dir / "gt"
        tilt_root = (
            (self.root_dir / "tilt")
            if (self.root_dir / "tilt").exists()
            else (self.root_dir / "tilt_only")
        )

        if deg_root.exists() and gt_root.exists():
            for entry in sorted(deg_root.iterdir()):
                # Case A: entry is a video file (e.g., `turb/0001.mp4`) or clip folder (`turb/0001/`)
                is_valid_entry = entry.is_dir() or (
                    entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS
                )
                if not is_valid_entry:
                    continue

                stem = entry.stem if entry.is_file() else entry.name
                gt_match = self._find_matching_entry(gt_root, entry.name, stem)
                if gt_match is not None:
                    tilt_match = (
                        self._find_matching_entry(tilt_root, entry.name, stem)
                        if tilt_root.exists()
                        else None
                    )
                    self.clips.append((entry, gt_match, tilt_match))

        self.simulator = PhysicsTurbulenceSimulator()

    @staticmethod
    def _find_matching_entry(
        parent_dir: Path, exact_name: str, stem: str
    ) -> Optional[Path]:
        """Finds a matching `.mp4` video, image file, or directory in `parent_dir`."""
        if (parent_dir / exact_name).exists():
            return parent_dir / exact_name
        if (parent_dir / stem).exists():
            return parent_dir / stem
        for ext in list(VIDEO_EXTENSIONS) + list(IMG_EXTENSIONS):
            candidate = parent_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

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

    def _load_entry_sequence(
        self,
        entry_path: Path,
        start_frame: int,
        indices: List[int],
    ) -> torch.Tensor:
        """Loads `num_frames` from either a `.mp4` video file, a static image, or a folder of frames."""
        if entry_path.is_file():
            ext = entry_path.suffix.lower()
            if ext in VIDEO_EXTENSIONS:
                return _load_video_clip_frames(
                    entry_path, start_frame=start_frame, num_frames=self.num_frames
                )
            if ext in IMG_EXTENSIONS:
                single = _load_rgb_tensor(entry_path)
                return single.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # Directory case: check if it contains images or a nested .mp4 video
        img_files = _list_image_files(entry_path)
        if len(img_files) == 1:
            single = _load_rgb_tensor(img_files[0])
            return single.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)
        if len(img_files) > 1:
            return torch.stack(
                [_load_rgb_tensor(img_files[i % len(img_files)]) for i in indices],
                dim=0,
            )

        vid_files = _list_video_files(entry_path)
        if len(vid_files) > 0:
            return _load_video_clip_frames(
                vid_files[0], start_frame=start_frame, num_frames=self.num_frames
            )
        raise RuntimeError(f"No valid video or image frames found in {entry_path}")

    def _get_entry_length(self, entry_path: Path) -> int:
        if entry_path.is_file():
            if entry_path.suffix.lower() in VIDEO_EXTENSIONS:
                return _get_video_frame_count(entry_path)
            return 1
        img_files = _list_image_files(entry_path)
        if len(img_files) > 0:
            return len(img_files)
        vid_files = _list_video_files(entry_path)
        if len(vid_files) > 0:
            return _get_video_frame_count(vid_files[0])
        return 0

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

        deg_entry, gt_entry, tilt_entry = self.clips[idx]
        deg_len = self._get_entry_length(deg_entry)
        gt_len = self._get_entry_length(gt_entry)
        total = deg_len if gt_len == 1 else min(deg_len, gt_len)
        total = max(1, total)

        if total >= self.num_frames:
            start = (
                int(torch.randint(0, total - self.num_frames + 1, (1,)).item())
                if self.is_train
                else 0
            )
            indices = list(range(start, start + self.num_frames))
        else:
            start = 0
            indices = [i % total for i in range(self.num_frames)]

        deg_seq = self._load_entry_sequence(deg_entry, start, indices)
        gt_seq = self._load_entry_sequence(gt_entry, start, indices)
        tilt_seq = (
            self._load_entry_sequence(tilt_entry, start, indices)
            if tilt_entry is not None
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

        clip_name = deg_entry.stem if deg_entry.is_file() else deg_entry.name
        return VideoBatch(
            degraded=deg_seq,
            gt=gt_seq,
            tilt_only=tilt_seq,
            meta={"clip_name": clip_name, "d_r0": 0.0},
        )
