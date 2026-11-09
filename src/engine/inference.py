"""High-Resolution Video Inference Engine with Temporal Sliding Window & 2D Hann Overlap-Tile."""

import math
from typing import Literal, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.tensor_ops import rearrange


class VideoInferenceEngine:
    """
    Production inference engine implementing Section 9 of `arquitetura.md`:
      1. Temporal Sliding Window (window_size W=12..36, temporal_stride S) with triangular
         temporal blending weights w_temp(tau) = 1 - |tau - (W-1)/2| / (W/2) to eliminate
         temporal flickering across window boundaries.
      2. Spatial Overlap-Tile (tile_size=512, tile_overlap=64) weighted by a 2D separable
         Hann window w_spat(x, y) = sin^2(pi*x/P_x) * sin^2(pi*y/P_y) so Full HD (1080p)
         or 4K videos run on an 8-12 GB GPU without seam artifacts or OOM.
         CRITICAL (§9.2 Step 2): Global turbulence strength Cn^2_hat is computed ONCE
         on the full-resolution frame before spatial tiling and shared across all tiles!
      3. Dual Output Mode:
         - `mode="video"`: Returns the complete restored sequence [T, 3, H, W].
         - `mode="forensic_snapshot"`: Returns a single ultra-sharp frame [1, 3, H, W]
           by selecting the highest-sharpness central frame for static recognition tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        window_size: int = 12,
        temporal_stride: int = 6,
        tile_size: int = 512,
        tile_overlap: int = 64,
        use_amp: bool = True,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.window_size = window_size
        self.temporal_stride = temporal_stride
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.use_amp = use_amp and (device.type == "cuda")
        self.amp_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

    def _build_temporal_weights(self, length: int, device: torch.device) -> torch.Tensor:
        """Triangular temporal ramp w_temp(tau) = 1 - |tau - (W-1)/2| / (W/2)."""
        if length == 1:
            return torch.ones(1, 1, 1, 1, device=device)
        tau = torch.arange(length, device=device, dtype=torch.float32)
        center = (length - 1) / 2.0
        half = max(1.0, length / 2.0)
        w = (1.0 - torch.abs(tau - center) / half).clamp(min=0.05)
        return w.view(length, 1, 1, 1)

    def _build_hann_window_2d(
        self, ph: int, pw: int, device: torch.device
    ) -> torch.Tensor:
        """2D separable raised-cosine (Hann) window for seamless spatial tile stitching."""
        wy = torch.sin(
            math.pi
            * (torch.arange(ph, device=device, dtype=torch.float32) + 0.5)
            / ph
        ).square()
        wx = torch.sin(
            math.pi
            * (torch.arange(pw, device=device, dtype=torch.float32) + 0.5)
            / pw
        ).square()
        w2d = torch.outer(wy, wx).clamp(min=1e-4)
        return w2d.view(1, 1, ph, pw)

    @staticmethod
    def _pad_to_multiple(
        video: torch.Tensor, mult: int = 16
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Pads [B, T, C, H, W] spatially to a multiple of `mult` using reflection."""
        b, t, _, h, w = video.shape
        pad_h = (mult - (h % mult)) % mult
        pad_w = (mult - (w % mult)) % mult
        if pad_h == 0 and pad_w == 0:
            return video, (h, w)
        flat = rearrange(video, "b t c h w -> (b t) c h w")
        padded = F.pad(flat, (0, pad_w, 0, pad_h), mode="reflect")
        return rearrange(padded, "(b t) c h w -> b t c h w", b=b, t=t), (h, w)

    @torch.no_grad()
    def _infer_clip_spatial_tiled(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Runs CWD-Net on a single temporal clip [1, T_w, 3, H, W], using spatial tiling
        with 2D Hann windowing if H or W exceeds `self.tile_size`.
        Computes `global_cn2` once across the full frame before tiling (§9.2 Step 2).
        """
        _, tw, c, h, w = clip.shape
        ts = self.tile_size
        ov = self.tile_overlap
        stride = ts - ov

        if h <= ts and w <= ts:
            clip_pad, (orig_h, orig_w) = self._pad_to_multiple(clip, mult=16)
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                out = self.model(clip_pad).restored.float()
            return out[:, :, :, :orig_h, :orig_w]

        # Step 2 of §9.2: Compute global scene turbulence strength Cn^2 ONCE on full frame
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        global_cn2 = (
            raw_model.estimate_global_cn2(clip)
            if hasattr(raw_model, "estimate_global_cn2")
            else None
        )

        # Spatial Overlap-Tile mode for high-resolution (e.g. 1080p) frames
        acc_canvas = torch.zeros(
            1, tw, c, h, w, device=self.device, dtype=torch.float32
        )
        weight_canvas = torch.zeros(
            1, 1, 1, h, w, device=self.device, dtype=torch.float32
        )
        hann_2d = self._build_hann_window_2d(ts, ts, self.device).unsqueeze(0)

        y_starts = list(range(0, max(1, h - ts + 1), stride))
        if y_starts[-1] + ts < h:
            y_starts.append(max(0, h - ts))

        x_starts = list(range(0, max(1, w - ts + 1), stride))
        if x_starts[-1] + ts < w:
            x_starts.append(max(0, w - ts))

        for y0 in y_starts:
            for x0 in x_starts:
                ph = min(ts, h - y0)
                pw = min(ts, w - x0)
                tile = clip[:, :, :, y0 : y0 + ph, x0 : x0 + pw]
                tile_pad, (th, tw_px) = self._pad_to_multiple(tile, mult=16)
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    pred_tile = (
                        self.model(tile_pad, precomputed_cn2=global_cn2)
                        .restored.float()[:, :, :, :th, :tw_px]
                    )

                w_patch = (
                    hann_2d
                    if (ph == ts and pw == ts)
                    else self._build_hann_window_2d(ph, pw, self.device).unsqueeze(0)
                )
                acc_canvas[:, :, :, y0 : y0 + ph, x0 : x0 + pw] += pred_tile * w_patch
                weight_canvas[:, :, :, y0 : y0 + ph, x0 : x0 + pw] += w_patch

        return acc_canvas / weight_canvas.clamp(min=1e-6)

    @torch.no_grad()
    def infer_video(
        self,
        video: torch.Tensor,
        mode: Literal["video", "forensic_snapshot"] = "video",
    ) -> torch.Tensor:
        """
        Args:
            video: Input degraded video [T, 3, H, W] or [1, T, 3, H, W] in [0, 1]
            mode: "video" returns [T, 3, H, W]; "forensic_snapshot" returns [1, 3, H, W]
        """
        if video.ndim == 4:
            video = video.unsqueeze(0)
        _, total_t, c, h, w = video.shape

        win = min(self.window_size, total_t)
        stride = max(1, min(self.temporal_stride, win))

        out_acc = torch.zeros(total_t, c, h, w, dtype=torch.float32)
        weight_acc = torch.zeros(total_t, 1, 1, 1, dtype=torch.float32)

        t_starts = list(range(0, max(1, total_t - win + 1), stride))
        if t_starts[-1] + win < total_t:
            t_starts.append(max(0, total_t - win))

        w_temp_gpu = self._build_temporal_weights(win, self.device)
        w_temp_cpu = w_temp_gpu.cpu()

        for t0 in t_starts:
            clip = video[:, t0 : t0 + win].to(self.device, non_blocking=True)
            restored_clip = self._infer_clip_spatial_tiled(clip).squeeze(0).cpu()
            out_acc[t0 : t0 + win] += restored_clip * w_temp_cpu
            weight_acc[t0 : t0 + win] += w_temp_cpu

        restored_video = (out_acc / weight_acc.clamp(min=1e-6)).clamp(0.0, 1.0)

        if mode == "forensic_snapshot":
            gx = restored_video[:, :, :, 1:] - restored_video[:, :, :, :-1]
            gy = restored_video[:, :, 1:, :] - restored_video[:, :, :-1, :]
            sharp_scores = gx.abs().mean(dim=(1, 2, 3)) + gy.abs().mean(dim=(1, 2, 3))
            best_idx = int(torch.argmax(sharp_scores).item())
            return restored_video[best_idx : best_idx + 1]

        return restored_video
