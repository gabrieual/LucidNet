"""2-Phase Curriculum Composite Loss for CWD-Net (§3 of `arquitetura.md`)."""

from typing import Dict, Tuple
import torch

from src.contracts.batch import ModelOutput, VideoBatch
from src.contracts.interfaces import TrainingLossWrapper
from src.losses.charbonnier import CharbonnierLoss, TotalVariationLoss
from src.losses.fft_loss import ComplexFFTLoss
from src.losses.cw_ssim import ComplexWaveletSSIMLoss
from src.models.blocks.daab import flow_warp
from src.utils.tensor_ops import rearrange


class CWDCompositeLoss(TrainingLossWrapper):
    """
    Implements the 2-Phase Curriculum Loss specified in Section 3 of `arquitetura.md`:
      - Phase 1 (First 70% of iterations):
          L_phase1 = alpha_tilt * L_tilt(W(I_tilt, T_hat^{-1}), J)
                   + alpha_char * L_char(J_hat, J)
                   + lambda_fft * L_FFT(J_hat, J)
                   + lambda_tv  * L_TV(T_hat^{-1})
        Default Phase 1 weights:
          alpha_tilt = 0.2, alpha_char = 0.8, lambda_fft = 0.05, lambda_tv = 0.01.
      - Phase 2 (Last 30% of iterations):
          L_phase2 = L_phase1
                   + 0.10 * (1 - CW_SSIM(J_hat_center, J_center))
                   + 0.02 * L_LPIPS(J_hat_center, J_center)
        where `center` selects the 4 central frames of the clip.
    """

    def __init__(
        self,
        lambda_char: float = 0.8,
        lambda_tilt: float = 0.2,
        lambda_fft: float = 0.05,
        lambda_tv: float = 0.01,
        lambda_cw_ssim: float = 0.1,
        lambda_lpips: float = 0.02,
        phase2_start_ratio: float = 0.70,
        total_iters: int = 200000,
        charbonnier_eps: float = 1e-3,
        use_lpips: bool = False,
    ) -> None:
        super().__init__()
        self.lambda_char = lambda_char
        self.lambda_tilt = lambda_tilt
        self.lambda_fft = lambda_fft
        self.lambda_tv = lambda_tv
        self.lambda_cw_ssim = lambda_cw_ssim
        self.lambda_lpips = lambda_lpips
        self.phase2_start_iter = int(total_iters * phase2_start_ratio)
        self.use_lpips = use_lpips

        self.charbonnier = CharbonnierLoss(eps=charbonnier_eps)
        self.tv_loss = TotalVariationLoss()
        self.fft_loss = ComplexFFTLoss()
        self.cw_ssim_loss = ComplexWaveletSSIMLoss()

        self._lpips_net = None
        if use_lpips:
            try:
                import lpips  # type: ignore

                self._lpips_net = lpips.LPIPS(net="vgg")
                for p in self._lpips_net.parameters():
                    p.requires_grad = False
            except ImportError:
                self._lpips_net = None

    @staticmethod
    def _slice_central_frames(
        pred: torch.Tensor, target: torch.Tensor, max_frames: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selects the central `max_frames` frames of a [B, T, C, H, W] clip (§3 Phase 2)."""
        t = pred.shape[1]
        k = min(max_frames, t)
        start = max(0, (t - k) // 2)
        return pred[:, start : start + k], target[:, start : start + k]

    def _compute_central_lpips(
        self, pred_center: torch.Tensor, target_center: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates LPIPS on the central 4 frames of the clip."""
        if self._lpips_net is None:
            return torch.tensor(0.0, device=pred_center.device)
        pred_flat = (
            rearrange(pred_center, "b t c h w -> (b t) c h w") * 2.0 - 1.0
        )
        target_flat = (
            rearrange(target_center, "b t c h w -> (b t) c h w") * 2.0 - 1.0
        )
        return self._lpips_net(pred_flat, target_flat).mean()

    def forward(
        self,
        output: ModelOutput,
        batch: VideoBatch,
        current_iter: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if batch.gt is None:
            raise ValueError(
                "Ground-truth tensor `batch.gt` is required for loss calculation."
            )

        j_hat = output.restored
        j_gt = batch.gt
        b, t, _, _, _ = j_hat.shape

        # 1. Charbonnier reconstruction loss L_char(J_hat, J)
        l_char = self.charbonnier(j_hat, j_gt)

        # 2. Geometric Tilt Rectification loss:
        #    L_tilt = Charbonnier( W(I_tilt, T_hat^{-1}), J_gt ) (§3 Eq. 1)
        if batch.tilt_only is not None:
            tilt_only_flat = rearrange(batch.tilt_only, "b t c h w -> (b t) c h w")
            inv_tilt_flat = rearrange(
                output.inverse_tilt_field, "b t c h w -> (b t) c h w"
            )
            rectified_tilt_flat = flow_warp(tilt_only_flat, inv_tilt_flat)
            rectified_tilt = rearrange(
                rectified_tilt_flat, "(b t) c h w -> b t c h w", b=b, t=t
            )
            l_tilt = self.charbonnier(rectified_tilt, j_gt)
        else:
            # Fallback if dataset does not provide separate I_tilt: compare W(I, T_hat^{-1}) with J_gt
            l_tilt = self.charbonnier(output.rectified_rgb, j_gt)

        # 3. Complex Fourier Loss L_FFT(J_hat, J)
        l_fft = self.fft_loss(j_hat, j_gt)

        # 4. Spatial smoothness regularization on the inverse tilt field L_TV(T_hat^{-1}) (§3 Eq. 4)
        l_tv = self.tv_loss(output.inverse_tilt_field)

        total_loss = (
            self.lambda_char * l_char
            + self.lambda_tilt * l_tilt
            + self.lambda_fft * l_fft
            + self.lambda_tv * l_tv
        )

        loss_dict: Dict[str, float] = {
            "loss/charbonnier": float(l_char.detach().item()),
            "loss/tilt": float(l_tilt.detach().item()),
            "loss/fft": float(l_fft.detach().item()),
            "loss/tv": float(l_tv.detach().item()),
            "loss/phase": 2.0 if current_iter >= self.phase2_start_iter else 1.0,
        }

        if current_iter >= self.phase2_start_iter:
            j_hat_center, j_gt_center = self._slice_central_frames(
                j_hat, j_gt, max_frames=4
            )
            l_cw = self.cw_ssim_loss(j_hat_center, j_gt_center)
            total_loss = total_loss + self.lambda_cw_ssim * l_cw
            loss_dict["loss/cw_ssim"] = float(l_cw.detach().item())

            if self.use_lpips and self.lambda_lpips > 0.0:
                l_lpips = self._compute_central_lpips(j_hat_center, j_gt_center)
                total_loss = total_loss + self.lambda_lpips * l_lpips
                loss_dict["loss/lpips"] = float(l_lpips.detach().item())

        loss_dict["loss/total"] = float(total_loss.detach().item())
        return total_loss, loss_dict
