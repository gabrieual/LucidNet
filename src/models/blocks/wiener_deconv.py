"""Block 3: Multi-Channel Blind Wiener Deconvolution in 2D-FFT (Numerically Shielded)."""

from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.contracts.interfaces import BaseDeconvBlock
from src.models.registry import DECONV_REGISTRY
from src.models.blocks.nafblock import SimpleGate


@DECONV_REGISTRY.register("wiener_fft_multichannel")
class MultiChannelWienerDeconv2D(BaseDeconvBlock):
    """
    Implements Section 2.4 of `arquitetura.md` (Explicit Physics-Driven Blind Wiener
    Deconvolution in 2D Fourier Domain):
      1. Upsamples fused features from scale 1/4 (in_channels=128) to scale 1/2
         (out_channels=64) via PixelShuffle.
      2. Estimates M=64 spatially-varying PSF basis kernels K_hat_m (15x15)
         constrained by Softmax (K_hat >= 0, sum(K_hat) = 1) and noise-to-signal ratios
         gamma_m >= eps_min (0.01) via Softplus.
      3. Performs exact 2D-FFT Wiener inversion with:
           - Reflective boundary padding (8 px) to eliminate Gibbs boundary ringing
           - Circular phase-centering roll (-k//2, -k//2) to guarantee zero spatial phase shift
           - Unscaled standard FFT (norm="backward") satisfying the Convolution Theorem
           - Forced float32 precision inside AMP contexts
           - True Dirac delta initialization (center logit = 12.0 -> >99.8% center energy)
           - Zero-initialized residual gate beta_deconv = 0 for stable training start
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 64,
        kernel_size: int = 15,
        eps_min: float = 0.01,
        pad_size: int = 8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps_min = eps_min
        self.pad_size = pad_size

        # 3.1 Upsample from 1/4 (128 ch) to 1/2 (64 ch) scale using PixelShuffle
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )

        # 3.2 Lightweight PSF & SNR Predictor:
        # Conv3x3(64 -> 64, stride=2) -> SimpleGate -> AdaptiveAvgPool2d(1) -> MLP
        self.psf_predictor = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            SimpleGate(),  # halves channels to out_channels // 2
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(out_channels // 2, out_channels),
            nn.GELU(),
        )
        self.kernel_head = nn.Linear(
            out_channels, out_channels * kernel_size * kernel_size
        )
        self.gamma_head = nn.Linear(out_channels, out_channels)

        # Initialize PSF logits so initial kernel after Softmax over 225 elements
        # is a true centered Dirac delta (exp(12) / (exp(12) + 224) = 0.9986 at center (7,7))
        nn.init.zeros_(self.kernel_head.weight)
        init_bias = torch.zeros(out_channels, kernel_size, kernel_size)
        center = kernel_size // 2
        init_bias[:, center, center] = 12.0
        self.kernel_head.bias.data.copy_(init_bias.view(-1))

        nn.init.zeros_(self.gamma_head.weight)
        nn.init.constant_(self.gamma_head.bias, -1.0)

        self.post_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        # Zero-initialized residual scaling parameter beta_deconv
        self.beta_deconv = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def _wiener_fft_step(
        self,
        f_up: torch.Tensor,
        psf_kernels: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Executes exact 2D-FFT Wiener deconvolution in float32.
        Args:
            f_up: [N, M, H, W]
            psf_kernels: [N, M, K, K] non-negative, unit-sum kernels
            gamma: [N, M, 1, 1] positive SNR regularization >= eps_min
        """
        n, m, h, w = f_up.shape
        p = self.pad_size
        k = self.kernel_size
        half_k = k // 2

        # Step A: Reflective padding to suppress periodic FFT boundary ringing
        f_pad = F.pad(f_up.float(), (p, p, p, p), mode="reflect")
        hp, wp = f_pad.shape[-2], f_pad.shape[-1]

        # Step B: Zero-pad PSF to (hp, wp) and circularly shift center (half_k, half_k) to (0, 0)
        k_pad = F.pad(
            psf_kernels.float(), (0, wp - k, 0, hp - k), mode="constant", value=0.0
        )
        k_shifted = torch.roll(k_pad, shifts=(-half_k, -half_k), dims=(-2, -1))

        # Step C: 2D Real FFT (standard norm="backward" so K_fft(0,0) = sum(K) = 1.0)
        x_fft = torch.fft.rfft2(f_pad)
        k_fft = torch.fft.rfft2(k_shifted)

        # Wiener Filter G(w) = conj(K(w)) / (|K(w)|^2 + gamma)
        k_mag_sq = (k_fft.real.square() + k_fft.imag.square()).clamp_min(1e-8)
        wiener_filter = torch.conj(k_fft) / (k_mag_sq + gamma.float())

        y_fft = wiener_filter * x_fft
        f_wiener_pad = torch.fft.irfft2(y_fft, s=(hp, wp))

        # Crop reflective padding back to (H, W)
        return f_wiener_pad[:, :, p : p + h, p : p + w]

    def forward(
        self, f_lucky: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        f_up = self.upsample(f_lucky)  # [N, 64, H/2, W/2]
        n = f_up.shape[0]
        m = self.out_channels
        k = self.kernel_size

        latent = self.psf_predictor(f_up)  # [N, M]
        kernel_logits = self.kernel_head(latent).view(n, m, k * k)
        psf_kernels = torch.softmax(kernel_logits.float(), dim=-1).view(n, m, k, k)

        gamma = self.eps_min + F.softplus(self.gamma_head(latent).float()).view(
            n, m, 1, 1
        )

        f_wiener = self._wiener_fft_step(f_up, psf_kernels, gamma).to(f_up.dtype)
        f_deconv = f_up + self.beta_deconv * self.post_conv(f_wiener)

        aux = {
            "psf_kernels": psf_kernels.detach(),
            "snr_gamma": gamma.detach(),
            "beta_deconv": self.beta_deconv.detach().abs().mean(),
        }
        return f_deconv, aux


@DECONV_REGISTRY.register("identity_upsample")
class IdentityUpsampleDeconv(BaseDeconvBlock):
    """
    Ablation counterpart for Block 3 (Variants A, B, C, D, and E-ZeroWiener):
    Performs the exact same PixelShuffle upsampling from 1/4 (128 ch) to 1/2 (64 ch) scale
    without Wiener deconvolution, keeping tensor shapes 100% compatible (Liskov Substitution).
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 64,
        kernel_size: int = 15,
        eps_min: float = 0.01,
        pad_size: int = 8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(
        self, f_lucky: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        f_up = self.upsample(f_lucky)
        return f_up, {
            "psf_kernels": None,
            "snr_gamma": None,
            "beta_deconv": torch.tensor(0.0, device=f_lucky.device),
        }
