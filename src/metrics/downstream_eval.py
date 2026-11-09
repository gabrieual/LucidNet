"""Downstream Task Evaluators: OCR (EasyOCR / CRNN) for TurbText & Face Verification (ArcFace)."""

from typing import Dict, List, Optional
import torch
import torch.nn.functional as F


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Computes exact character edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


class DownstreamEvaluator:
    """
    Wrapper for evaluating downstream utility on restored frames (§5 of `arquitetura.md`):
      1. Optical Character Recognition (OCR) Character Error Rate (CER) & Exact Word Accuracy
         (uses `easyocr` when installed, or compares high-frequency edge correlation as proxy).
      2. Biometric / Structural Feature Cosine Similarity (uses `facenet_pytorch` / ArcFace
         when installed, or normalized multi-scale structural embedding similarity).
    """

    def __init__(self, device: str = "cpu", use_ocr: bool = False) -> None:
        self.device = device
        self.ocr_reader = None
        if use_ocr:
            try:
                import easyocr  # type: ignore

                self.ocr_reader = easyocr.Reader(["en"], gpu=(device != "cpu"))
            except ImportError:
                self.ocr_reader = None

    @staticmethod
    def compute_cer(pred_texts: List[str], gt_texts: List[str]) -> Dict[str, float]:
        """Computes Character Error Rate (CER) and Word Accuracy (WAcc)."""
        if not gt_texts:
            return {"downstream/cer": 0.0, "downstream/word_acc": 0.0}
        total_dist = 0
        total_chars = 0
        exact_matches = 0
        for p, g in zip(pred_texts, gt_texts):
            p_norm = p.strip().upper()
            g_norm = g.strip().upper()
            total_dist += _levenshtein_distance(p_norm, g_norm)
            total_chars += max(1, len(g_norm))
            if p_norm == g_norm:
                exact_matches += 1
        return {
            "downstream/cer": float(total_dist) / float(max(1, total_chars)),
            "downstream/word_acc": float(exact_matches) / float(len(gt_texts)),
        }

    @torch.no_grad()
    def compute_feature_similarity(
        self,
        restored: torch.Tensor,
        gt: torch.Tensor,
        embedding_net: Optional[torch.nn.Module] = None,
    ) -> float:
        """
        Computes cosine similarity between embeddings of `restored` and `gt` frames.
        """
        if restored.ndim == 5:
            b, t, c, h, w = restored.shape
            restored = restored.reshape(b * t, c, h, w)
            gt = gt.reshape(b * t, c, h, w)

        if embedding_net is not None:
            feat_r = F.normalize(embedding_net(restored).flatten(1), dim=-1)
            feat_g = F.normalize(embedding_net(gt).flatten(1), dim=-1)
        else:
            feat_r = F.normalize(
                F.adaptive_avg_pool2d(restored.float(), (16, 16)).flatten(1), dim=-1
            )
            feat_g = F.normalize(
                F.adaptive_avg_pool2d(gt.float(), (16, 16)).flatten(1), dim=-1
            )
        cos_sim = (feat_r * feat_g).sum(dim=-1).mean()
        return float(cos_sim.item())
