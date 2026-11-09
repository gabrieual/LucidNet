"""Classical Spectral & Directional Wavelet/Gabor Sharpness Priority Map (CLEAR & proposta.pdf)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.tensor_ops import rearrange


class SpectralSharpnessPrior(nn.Module):
    """
    Implements Section 2.1.4 of `arquitetura.md`:
    Computes the classical Lucky Imaging priority map P_t(x, y) in R^{B x T x 1 x H/4 x W/4}
    by MULTIPLYING:
      1. Local band-pass energy G_t(x, y):
         - Sum of magnitudes across 6 directional complex Gabor/DT-CWT sub-bands
           at (+/-15°, +/-45°, +/-75°)
         - Combined with 8x8 block 2D-FFT mid-frequency band-pass energy
           (discarding DC and ultra-high sensor thermal noise frequencies).
      2. CLEAR outlier similarity penalty (CLEAR.jsp, Eq. 2):
         S_t(x, y) = 1 / (lambda_S + ||I_t - I_{ref_init, t}||_2^2)
         P_t(x, y) = G_t(x, y) * S_t(x, y)
    """

    def __init__(
        self,
        patch_size: int = 8,
        lambda_s: float = 0.02,
        outlier_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.lambda_s = lambda_s
        self.outlier_weight = outlier_weight

        # Build 6 complex directional filters at angles +/-15, +/-45, +/-75 degrees (matching DT-CWT)
        angles_deg = [-75.0, -45.0, -15.0, 15.0, 45.0, 75.0]
        real_kernels, imag_kernels = [], []
        ksize = 7
        sigma = 1.6
        wavelength = 3.0
        coords = torch.arange(-(ksize // 2), ksize // 2 + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        for deg in angles_deg:
            theta = math.radians(deg)
            x_theta = xx * math.cos(theta) + yy * math.sin(theta)
            y_theta = -xx * math.sin(theta) + yy * math.cos(theta)
            gauss = torch.exp(-0.5 * (x_theta.square() + y_theta.square()) / (sigma ** 2))
            real_k = gauss * torch.cos(2.0 * math.pi * x_theta / wavelength)
            imag_k = gauss * torch.sin(2.0 * math.pi * x_theta / wavelength)
            real_k = real_k - real_k.mean()
            imag_k = imag_k - imag_k.mean()
            real_kernels.append(real_k)
            imag_kernels.append(imag_k)

        self.register_buffer(
            "gabor_real", torch.stack(real_kernels, dim=0).unsqueeze(1)
        )  # [6, 1, 7, 7]
        self.register_buffer(
            "gabor_imag", torch.stack(imag_kernels, dim=0).unsqueeze(1)
        )  # [6, 1, 7, 7]

        # Build 8x8 2D-FFT mid-frequency band-pass mask (excludes DC (0,0) and extreme Nyquist corners)
        fy = torch.fft.fftfreq(patch_size)
        fx = torch.fft.rfftfreq(patch_size)
        gy, gx = torch.meshgrid(fy, fx, indexing="ij")
        rad = torch.sqrt(gx.square() + gy.square())
        # Pass mid-band frequencies (0.08 <= f <= 0.45 cycles/pixel), rejecting DC & thermal noise
        bp_mask = ((rad >= 0.08) & (rad <= 0.45)).float()
        self.register_buffer("fft_bandpass_mask", bp_mask.view(1, 1, patch_size, -1))

    def _compute_8x8_fft_bandpass(
        self, lum: torch.Tensor, out_h: int, out_w: int
    ) -> torch.Tensor:
        """Computes local 8x8 block 2D-FFT band-pass energy resized to (out_h, out_w)."""
        n, _, h, w = lum.shape
        ps = self.patch_size
        pad_h = (ps - (h % ps)) % ps
        pad_w = (ps - (w % ps)) % ps
        lum_pad = F.pad(lum, (0, pad_w, 0, pad_h), mode="reflect")
        hp, wp = lum_pad.shape[-2], lum_pad.shape[-1]
        nh, nw = hp // ps, wp // ps

        blocks = lum_pad.reshape(n, 1, nh, ps, nw, ps).permute(0, 2, 4, 1, 3, 5)
        blocks_flat = blocks.reshape(n * nh * nw, 1, ps, ps)
        spec = torch.fft.rfft2(blocks_flat)
        mag = torch.sqrt(spec.real.square() + spec.imag.square() + 1e-8)
        bp_energy = (mag * self.fft_bandpass_mask).mean(dim=(-2, -1))  # [N*nh*nw, 1]
        bp_map = bp_energy.view(n, 1, nh, nw)
        return F.interpolate(bp_map, size=(out_h, out_w), mode="bilinear", align_corners=False)

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(
        self,
        video: torch.Tensor,
        reference_frames: torch.Tensor,
        target_scale: int = 4,
    ) -> torch.Tensor:
        """
        Args:
            video: [B, T, C, H, W]
            reference_frames: [B, T, C, H, W]
            target_scale: Downsampling factor for output priority map (default 4 -> H//4, W//4)
        Returns:
            priority_map: [B, T, 1, H//target_scale, W//target_scale] normalized sharpness priority
        """
        orig_dtype = video.dtype
        video_f32 = video.float()
        ref_f32 = reference_frames.float()
        b, t, _, h, w = video_f32.shape
        out_h, out_w = h // target_scale, w // target_scale

        lum = rearrange(video_f32.mean(dim=2, keepdim=True), "b t c h w -> (b t) c h w")

        # 1a. 6-orientation complex Gabor/Wavelet directional magnitude sum
        resp_r = F.conv2d(lum, self.gabor_real, padding=3)  # [B*T, 6, H, W]
        resp_i = F.conv2d(lum, self.gabor_imag, padding=3)  # [B*T, 6, H, W]
        mag_6dir = torch.sqrt(resp_r.square() + resp_i.square() + 1e-8)
        wavelet_energy = mag_6dir.sum(dim=1, keepdim=True)  # [B*T, 1, H, W]
        wavelet_low = F.avg_pool2d(
            wavelet_energy, kernel_size=target_scale, stride=target_scale
        )

        # 1b. 8x8 block 2D-FFT band-pass energy
        fft_low = self._compute_8x8_fft_bandpass(lum, out_h, out_w)

        sharpness_g = rearrange(
            wavelet_low + fft_low, "(b t) c h w -> b t c h w", b=b, t=t
        )
        mean_g = sharpness_g.mean(dim=1, keepdim=True) + 1e-6
        g_norm = sharpness_g / (mean_g + sharpness_g)

        if self.outlier_weight <= 0.0:
            return g_norm.to(orig_dtype)

        # 2. Similarity term S_t = 1 / (lambda_S + ||I_t - I_{ref_init, t}||_2^2) (CLEAR Eq. 2)
        diff_sq = (video_f32 - ref_f32).square().sum(dim=2, keepdim=True)  # [B, T, 1, H, W]
        diff_low = F.avg_pool2d(
            rearrange(diff_sq, "b t c h w -> (b t) c h w"),
            kernel_size=target_scale,
            stride=target_scale,
        )
        diff_low = rearrange(diff_low, "(b t) c h w -> b t c h w", b=b, t=t)

        similarity_s = 1.0 / (self.lambda_s + diff_low)
        mean_s = similarity_s.mean(dim=1, keepdim=True) + 1e-6
        s_norm = similarity_s / (mean_s + similarity_s)

        # Multiplicative combination P_t = G_t * S_t (§2.1.4 of arquitetura.md)
        priority = g_norm * (s_norm ** self.outlier_weight)
        return priority.to(orig_dtype)
