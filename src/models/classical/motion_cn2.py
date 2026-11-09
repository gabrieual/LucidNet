"""Physical Turbulence Strength (Cn²) Estimator and AOF Motion Segmentation Mask."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.tensor_ops import rearrange


class MotionCn2Estimator(nn.Module):
    """
    Implements Section 2.1.3 of `arquitetura.md`:
      1. Relative turbulence strength estimate:
             Cn^2_hat = Mean_{x,y}(Var_t(I)) / (Mean_{x,y,t}(||grad_{x,y} I_t||_2) + eps)
      2. Average Optical Flow (AOF) across neighbor frames k in [-K, K]:
             AOF_t(x,y) = (1 / (1 + Cn^2_hat)) * || (1 / 2K) * sum_{k=-K, k!=0}^K u_{t -> t+k}(x,y) ||_2
         where u_{t -> t+k} = (u_x, u_y) is the dense Lucas-Kanade optical flow vector field.
         Zero-mean random turbulence vectors cancel out inside the vector sum (||sum u_turb|| ~ 0),
         whereas coherent moving objects accumulate constructively (||sum u_obj|| >> 0).
      3. Smooth motion mask:
             M_t(x,y) = Sigmoid( kappa * (AOF_t(x,y) - tau_mov) )
    """

    def __init__(
        self,
        window_size: int = 5,
        motion_threshold: float = 0.08,
        sharpness_k: float = 25.0,
        lk_win_size: int = 5,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.motion_threshold = motion_threshold
        self.sharpness_k = sharpness_k
        self.lk_win_size = lk_win_size

        sobel_x = (
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        )
        sobel_y = (
            torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) / 8.0
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _spatial_gradients(
        self, lum: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute spatial gradients Ix, Iy and magnitude ||grad I||_2 on [N, 1, H, W]."""
        gx = F.conv2d(lum, self.sobel_x.to(lum.dtype), padding=1)
        gy = F.conv2d(lum, self.sobel_y.to(lum.dtype), padding=1)
        mag = torch.sqrt(gx.square() + gy.square() + 1e-8)
        return gx, gy, mag

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def estimate_cn2(self, video: torch.Tensor) -> torch.Tensor:
        """
        Computes global scene turbulence strength Cn^2_hat [B] on the full frame
        (used directly in `forward` or once before spatial tiling in `VideoInferenceEngine`).
        """
        video_f32 = video.float()
        b, t, _, h, w = video_f32.shape
        lum = video_f32.mean(dim=2, keepdim=True)  # [B, T, 1, H, W]
        flat_lum = rearrange(lum, "b t c h w -> (b t) c h w")
        _, _, grad_mag = self._spatial_gradients(flat_lum)
        grad_mag_seq = rearrange(grad_mag, "(b t) c h w -> b t c h w", b=b, t=t)

        temp_var = (
            lum.var(dim=1, unbiased=False) if t > 1 else torch.zeros_like(lum[:, 0])
        )
        mean_var = temp_var.mean(dim=(-3, -2, -1))  # [B]
        mean_grad = grad_mag_seq.mean(dim=(1, 2, 3, 4))  # [B]
        return mean_var / (mean_grad + 1e-4)

    def _lucas_kanade_flow(
        self,
        ix: torch.Tensor,
        iy: torch.Tensor,
        it: torch.Tensor,
        s_xx: torch.Tensor,
        s_yy: torch.Tensor,
        s_xy: torch.Tensor,
        inv_det: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized 2D Lucas-Kanade optical flow solver for a pair of frames:
            [s_xx  s_xy] [u_x] = -[s_xt]
            [s_xy  s_yy] [u_y]   -[s_yt]
        Returns flow u = (u_x, u_y) of shape [B, 2, H, W].
        """
        win = self.lk_win_size
        pad = win // 2
        s_xt = F.avg_pool2d(ix * it, kernel_size=win, stride=1, padding=pad)
        s_yt = F.avg_pool2d(iy * it, kernel_size=win, stride=1, padding=pad)

        ux = (-s_yy * s_xt + s_xy * s_yt) * inv_det
        uy = (s_xy * s_xt - s_xx * s_yt) * inv_det
        return torch.cat([ux, uy], dim=1).clamp(-10.0, 10.0)

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(
        self,
        video: torch.Tensor,
        frf_static: Optional[torch.Tensor] = None,
        precomputed_cn2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video: [B, T, C, H, W]
            frf_static: Optional [B, C, H, W] reference frame
            precomputed_cn2: Optional precomputed scene-wide Cn^2 [B] (from full frame before tiling)
        Returns:
            cn2_est: [B] relative turbulence intensity scalar per batch item
            motion_mask: [B, T, 1, H, W] in [0, 1] (0 = static background, 1 = moving object)
        """
        orig_dtype = video.dtype
        video_f32 = video.float()
        b, t, _, h, w = video_f32.shape
        lum = video_f32.mean(dim=2, keepdim=True)  # [B, T, 1, H, W]
        flat_lum = rearrange(lum, "b t c h w -> (b t) c h w")

        gx, gy, grad_mag = self._spatial_gradients(flat_lum)

        if precomputed_cn2 is not None:
            cn2_est = precomputed_cn2.to(device=video.device, dtype=torch.float32)
        else:
            grad_mag_seq = rearrange(grad_mag, "(b t) c h w -> b t c h w", b=b, t=t)
            temp_var = (
                lum.var(dim=1, unbiased=False) if t > 1 else torch.zeros_like(lum[:, 0])
            )
            mean_var = temp_var.mean(dim=(-3, -2, -1))
            mean_grad = grad_mag_seq.mean(dim=(1, 2, 3, 4))
            cn2_est = mean_var / (mean_grad + 1e-4)

        if t <= 2:
            zero_mask = torch.zeros((b, t, 1, h, w), device=video.device, dtype=orig_dtype)
            return cn2_est.to(orig_dtype), zero_mask

        # Precompute spatial structure tensor components for Lucas-Kanade optical flow
        win = self.lk_win_size
        pad = win // 2
        s_xx = F.avg_pool2d(gx * gx, kernel_size=win, stride=1, padding=pad) + 1e-3
        s_yy = F.avg_pool2d(gy * gy, kernel_size=win, stride=1, padding=pad) + 1e-3
        s_xy = F.avg_pool2d(gx * gy, kernel_size=win, stride=1, padding=pad)
        det = (s_xx * s_yy - s_xy * s_xy).clamp(min=1e-6)
        inv_det = 1.0 / det

        gx_seq = rearrange(gx, "(b t) c h w -> b t c h w", b=b, t=t)
        gy_seq = rearrange(gy, "(b t) c h w -> b t c h w", b=b, t=t)
        s_xx_seq = rearrange(s_xx, "(b t) c h w -> b t c h w", b=b, t=t)
        s_yy_seq = rearrange(s_yy, "(b t) c h w -> b t c h w", b=b, t=t)
        s_xy_seq = rearrange(s_xy, "(b t) c h w -> b t c h w", b=b, t=t)
        inv_det_seq = rearrange(inv_det, "(b t) c h w -> b t c h w", b=b, t=t)

        # Compute Average Optical Flow AOF_t(x,y) = (1/(1+Cn2)) * || (1/2K) * sum_{k=-K, k!=0}^K u_{t->t+k} ||_2
        half_k = max(1, self.window_size // 2)
        offsets = [k for k in range(-half_k, half_k + 1) if k != 0]
        num_offsets = float(len(offsets))

        aof_list = []
        for idx in range(t):
            vec_sum = torch.zeros((b, 2, h, w), device=video.device, dtype=torch.float32)
            ix_t = gx_seq[:, idx]
            iy_t = gy_seq[:, idx]
            sxx_t = s_xx_seq[:, idx]
            syy_t = s_yy_seq[:, idx]
            sxy_t = s_xy_seq[:, idx]
            idet_t = inv_det_seq[:, idx]
            lum_t = lum[:, idx]

            for k_off in offsets:
                nbr_idx = min(max(idx + k_off, 0), t - 1)
                it_k = (lum[:, nbr_idx] - lum_t) / float(abs(k_off))
                u_tk = self._lucas_kanade_flow(
                    ix_t, iy_t, it_k, sxx_t, syy_t, sxy_t, idet_t
                )
                # Normalize direction sign so forward (k>0) and backward (k<0) displacements align
                sign = 1.0 if k_off > 0 else -1.0
                vec_sum = vec_sum + sign * u_tk

            mean_flow_vec = vec_sum / num_offsets  # [B, 2, H, W]
            # Vector L2 magnitude || (1/2K) * sum u_{t->t+k} ||_2
            flow_mag = torch.sqrt(
                mean_flow_vec[:, 0:1].square() + mean_flow_vec[:, 1:2].square() + 1e-8
            )
            aof_list.append(flow_mag)

        aof_seq = torch.stack(aof_list, dim=1)  # [B, T, 1, H, W]
        cn2_factor = (1.0 / (1.0 + cn2_est)).view(b, 1, 1, 1, 1)
        aof_modulated = aof_seq * cn2_factor

        # Smooth spatially to form coherent object masks
        aof_smooth = F.avg_pool2d(
            rearrange(aof_modulated, "b t c h w -> (b t) c h w"),
            kernel_size=9,
            stride=1,
            padding=4,
        )
        aof_smooth = rearrange(aof_smooth, "(b t) c h w -> b t c h w", b=b, t=t)

        motion_mask = torch.sigmoid(
            self.sharpness_k * (aof_smooth - self.motion_threshold)
        )
        return cn2_est.to(orig_dtype), motion_mask.to(orig_dtype)
