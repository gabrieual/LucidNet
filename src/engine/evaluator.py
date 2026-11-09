"""Validation & Benchmark Evaluator with CSV and LaTeX Ablation Table Generation."""

import csv
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.metrics.optical_metrics import (
    MetricTracker,
    compute_cw_ssim,
    compute_psnr,
    compute_ssim,
)


class Evaluator:
    """
    Runs quantitative evaluation across paired turbulence test sets (ATSyn, HeatChamber, TUG)
    and exports results to both CSV and publication-ready LaTeX tables (§5 of `arquitetura.md`).
    Measures intermediate FRF PSNR, Tilt-Rectified PSNR, and Final Restored PSNR/SSIM/CW-SSIM.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @torch.no_grad()
    def evaluate_loader(
        self, model: nn.Module, loader: DataLoader
    ) -> Dict[str, float]:
        model = model.to(self.device).eval()
        tracker = MetricTracker()

        for batch in loader:
            batch = batch.to(self.device)
            output = model(batch.degraded)
            if batch.gt is not None:
                b, t, _, _, _ = batch.gt.shape
                frf_exp = output.frf_reference.unsqueeze(1).expand_as(batch.gt)
                tracker.update(
                    {
                        "PSNR": compute_psnr(output.restored, batch.gt),
                        "SSIM": compute_ssim(output.restored, batch.gt),
                        "CW-SSIM": compute_cw_ssim(output.restored, batch.gt),
                        "Rectified_PSNR": compute_psnr(output.rectified_rgb, batch.gt),
                        "FRF_PSNR": compute_psnr(frf_exp, batch.gt),
                    },
                    n=b,
                )

        return tracker.result()

    @staticmethod
    def export_ablation_tables(
        results_by_variant: Dict[str, Dict[str, float]],
        csv_path: str | Path,
        latex_path: Optional[str | Path] = None,
    ) -> str:
        """
        Exports a dictionary `{variant_name: {"PSNR": ..., "SSIM": ..., "CW-SSIM": ...}}`
        to CSV and returns/saves a formatted LaTeX `booktabs` table.
        """
        csv_out = Path(csv_path)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        metric_cols: List[str] = ["FRF_PSNR", "Rectified_PSNR", "PSNR", "SSIM", "CW-SSIM"]

        with csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Variant"] + metric_cols)
            for var_name, m in results_by_variant.items():
                writer.writerow(
                    [var_name] + [f"{m.get(col, 0.0):.4f}" for col in metric_cols]
                )

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Ablation Study of CWD-Net Variants (A $\to$ E)}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Variant & FRF PSNR & Rectified PSNR & PSNR (dB) $\uparrow$ & SSIM $\uparrow$ & CW-SSIM $\uparrow$ \\",
            r"\midrule",
        ]
        for var_name, m in results_by_variant.items():
            safe_name = var_name.replace("_", r"\_")
            row_vals = " & ".join(
                [
                    f"{m.get('FRF_PSNR', 0.0):.2f}",
                    f"{m.get('Rectified_PSNR', 0.0):.2f}",
                    f"{m.get('PSNR', 0.0):.2f}",
                    f"{m.get('SSIM', 0.0):.4f}",
                    f"{m.get('CW-SSIM', 0.0):.4f}",
                ]
            )
            lines.append(f"{safe_name} & {row_vals} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        latex_str = "\n".join(lines) + "\n"

        if latex_path is not None:
            tex_out = Path(latex_path)
            tex_out.parent.mkdir(parents=True, exist_ok=True)
            tex_out.write_text(latex_str, encoding="utf-8")

        return latex_str
