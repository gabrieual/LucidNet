"""Sanity-check test: Overfitting CWD-Net on a single synthetic turbulent clip."""

from pathlib import Path
import pytest

from src.utils.config_loader import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_overfit_single_batch() -> None:
    """
    Verifies that CWD-Net (Variant E) can overfit a single synthetic P2S turbulent clip
    over 15 gradient steps with strictly decreasing loss and finite gradients.
    """
    torch = pytest.importorskip("torch")
    from src.contracts.batch import VideoBatch
    from src.data.p2s_simulator import PhysicsTurbulenceSimulator
    from src.losses.composite_loss import CWDCompositeLoss
    from src.models.cwd_net import build_cwd_net
    from src.utils.reproducibility import seed_everything

    seed_everything(42, deterministic=True)
    cfg = load_config(PROJECT_ROOT / "configs/ablations/E_cwdnet_full.yaml")
    cfg["model"]["dims"] = {"c1": 16, "c2": 32, "c3": 48}
    cfg["model"]["decoder"]["num_blocks_s1"] = 1
    cfg["model"]["decoder"]["num_blocks_s2"] = 1

    model = build_cwd_net(cfg).train()
    criterion = CWDCompositeLoss(total_iters=20, use_lpips=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create a single deterministic clean clip [T=4, 3, 64, 64] and simulate turbulence
    t, c, h, w = 4, 3, 64, 64
    clean = torch.rand(t, c, h, w)
    simulator = PhysicsTurbulenceSimulator()
    degraded, tilt_only, _ = simulator(clean)

    batch = VideoBatch(
        degraded=degraded.unsqueeze(0),
        gt=clean.unsqueeze(0),
        tilt_only=tilt_only.unsqueeze(0),
    )

    initial_loss = None
    final_loss = None
    for step in range(12):
        optimizer.zero_grad()
        out = model(batch.degraded)
        loss, _ = criterion(out, batch, current_iter=step)
        assert torch.isfinite(loss), f"Non-finite loss at step {step}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
        optimizer.step()

        if step == 0:
            initial_loss = float(loss.item())
        final_loss = float(loss.item())

    assert initial_loss is not None and final_loss is not None
    assert final_loss < initial_loss, (
        f"Expected model to reduce loss on single batch, got {initial_loss:.4f} -> {final_loss:.4f}"
    )
