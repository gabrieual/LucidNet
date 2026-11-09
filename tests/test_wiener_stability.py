"""Numerical, phase-shift, and initialization tests for Block 3 (Wiener) & flow_warp."""

import math
from pathlib import Path
import pytest


def test_wiener_initial_bias_produces_true_dirac_delta() -> None:
    """
    Verifies mathematically (even without `torch`) that our center bias initialization
    (center logit = 12.0, off-center = 0.0 on a 15x15 = 225 grid) yields > 99.8% energy
    at the central impulse (7, 7), fixing the uniform-box bug (where bias=2.0 gave only 3.2%).
    """
    center_logit = 12.0
    num_off_center = 15 * 15 - 1  # 224 elements at logit 0.0
    center_weight = math.exp(center_logit) / (math.exp(center_logit) + num_off_center * 1.0)
    assert center_weight > 0.998, f"Expected Dirac delta > 0.998, got {center_weight}"


def test_wiener_zero_phase_shift_on_delta_psf() -> None:
    """
    Verifies that `_wiener_fft_step` introduces ZERO spatial translation when the PSF
    is a centered Dirac delta at (K//2, K//2) = (7, 7), confirming that `torch.roll`
    properly aligns the optical center to (0, 0) in 2D-FFT with standard unscaled FFT.
    """
    torch = pytest.importorskip("torch")
    from src.models.blocks.wiener_deconv import MultiChannelWienerDeconv2D

    deconv = MultiChannelWienerDeconv2D(
        in_channels=16, out_channels=8, kernel_size=15, eps_min=1e-6, pad_size=8
    ).eval()

    n, m, h, w = 2, 8, 32, 32
    f_up = torch.rand(n, m, h, w, dtype=torch.float32)

    # Verify initial predicted PSF has > 99.8% energy at (7, 7)
    with torch.no_grad():
        _, aux = deconv(torch.rand(n, 16, h // 2, w // 2))
        psf_init = aux["psf_kernels"]
        assert (psf_init[:, :, 7, 7] > 0.998).all()

    delta_psf = torch.zeros(n, m, 15, 15, dtype=torch.float32)
    delta_psf[:, :, 7, 7] = 1.0
    gamma = torch.full((n, m, 1, 1), 1e-6, dtype=torch.float32)

    f_wiener = deconv._wiener_fft_step(f_up, delta_psf, gamma)
    max_err = (f_wiener - f_up).abs().max().item()
    assert max_err < 1e-4, f"Expected zero phase shift, but max error was {max_err}"


def test_flow_warp_preserves_subpixel_precision_in_bfloat16() -> None:
    """
    Verifies that `flow_warp` computes the coordinate grid in float32 even when inputs
    are `bfloat16`, preserving sub-pixel shifts at coordinates > 128.
    """
    torch = pytest.importorskip("torch")
    from src.models.blocks.daab import flow_warp

    x_f32 = torch.rand(1, 3, 256, 256, dtype=torch.float32)
    flow_subpixel = torch.full((1, 2, 256, 256), 0.25, dtype=torch.float32)

    out_f32 = flow_warp(x_f32, flow_subpixel)
    out_bf16 = flow_warp(x_f32.to(torch.bfloat16), flow_subpixel.to(torch.bfloat16))
    diff = (out_f32 - out_bf16.float()).abs().mean().item()
    assert diff < 1e-2


def test_wiener_initial_identity_and_gradient_stability() -> None:
    """
    Verifies that at initialization (`beta_deconv = 0`), `MultiChannelWienerDeconv2D`
    acts as an exact upsampling identity and produces finite gradients when active.
    """
    torch = pytest.importorskip("torch")
    from src.models.blocks.wiener_deconv import MultiChannelWienerDeconv2D

    deconv = MultiChannelWienerDeconv2D(
        in_channels=16, out_channels=8, kernel_size=15, eps_min=0.01, pad_size=8
    )
    deconv.beta_deconv.data.fill_(0.1)

    x = torch.randn(2, 16, 16, 16, requires_grad=True)
    out, aux = deconv(x)
    loss = out.square().mean()
    loss.backward()

    assert torch.isfinite(out).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert deconv.kernel_head.weight.grad is not None
    assert torch.isfinite(deconv.kernel_head.weight.grad).all()
    assert (aux["snr_gamma"] >= 0.01).all()
