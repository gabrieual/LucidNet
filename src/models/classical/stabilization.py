"""2D-FFT Global Camera Shake Stabilization (Zero-Mean Centroid Anchored)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.tensor_ops import rearrange


class FFTGlobalStabilizer(nn.Module):
    """
    Compensates global rigid camera shake across a sequence [B, T, C, H, W]
    using GPU-accelerated 2D cross-correlation in the Fourier domain (§2.1.1 of arquitetura.md).
    Anchors shifts to the temporal centroid (zero-mean across T) so that zero-mean
    atmospheric tilt in Frame 0 never biases the global coordinate system away from Ground-Truth.
    """

    def __init__(
        self,
        enabled: bool = True,
        stabilize_in_train: bool = False,
        max_shift_ratio: float = 0.15,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.stabilize_in_train = stabilize_in_train
        self.max_shift_ratio = max_shift_ratio

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, T, C, H, W]
        Returns:
            stabilized: [B, T, C, H, W]
        """
        if not self.enabled or video.shape[1] <= 1:
            return video
        if self.training and not self.stabilize_in_train:
            return video

        orig_dtype = video.dtype
        video_f32 = video.float()
        b, t, c, h, w = video_f32.shape

        # Convert to luminance [B, T, H, W] and subtract spatial mean
        lum = video_f32.mean(dim=2)
        lum = lum - lum.mean(dim=(-2, -1), keepdim=True)

        ref_fft = torch.fft.rfft2(lum[:, 0:1])  # [B, 1, H, W//2+1]
        cur_fft = torch.fft.rfft2(lum)  # [B, T, H, W//2+1]

        cross_power = ref_fft * torch.conj(cur_fft)
        corr = torch.fft.irfft2(cross_power, s=(h, w))  # [B, T, H, W]
        corr = torch.fft.fftshift(corr, dim=(-2, -1))

        # Restrict search window around center (H//2, W//2)
        cy, cx = h // 2, w // 2
        max_dy = max(1, int(h * self.max_shift_ratio))
        max_dx = max(1, int(w * self.max_shift_ratio))

        window = corr[
            :, :, cy - max_dy : cy + max_dy + 1, cx - max_dx : cx + max_dx + 1
        ]
        win_w = window.shape[-1]
        flat_idx = window.reshape(b, t, -1).argmax(dim=-1)  # [B, T]

        dy = ((flat_idx // win_w) - max_dy).float()  # [B, T]
        dx = ((flat_idx % win_w) - max_dx).float()  # [B, T]

        # Force frame 0 relative displacement to 0, then subtract temporal mean
        # so the clip stays anchored to the scene's zero-mean coordinate system
        dy[:, 0] = 0.0
        dx[:, 0] = 0.0
        dy = dy - dy.mean(dim=1, keepdim=True)
        dx = dx - dx.mean(dim=1, keepdim=True)

        # Apply translation via float32 normalized grid_sample
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=video.device, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, w, device=video.device, dtype=torch.float32),
            indexing="ij",
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)

        norm_dx = (2.0 * dx / max(w - 1, 1)).view(b, t, 1, 1)
        norm_dy = (2.0 * dy / max(h - 1, 1)).view(b, t, 1, 1)
        shift_grid = torch.stack([-norm_dx, -norm_dy], dim=-1)  # [B, T, 1, 1, 2]

        sampling_grid = (base_grid + shift_grid).reshape(b * t, h, w, 2)
        flat_video = rearrange(video_f32, "b t c h w -> (b t) c h w")
        stabilized = F.grid_sample(
            flat_video,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return rearrange(stabilized, "(b t) c h w -> b t c h w", b=b, t=t).to(orig_dtype)
