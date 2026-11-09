"""Explicit tensor rearrangement using `einops.rearrange` with a native PyTorch fallback."""

from typing import Any
import torch

try:
    from einops import rearrange as _einops_rearrange  # type: ignore

    HAS_EINOPS = True
except ImportError:
    _einops_rearrange = None
    HAS_EINOPS = False


def rearrange(tensor: torch.Tensor, pattern: str, **axes_lengths: int) -> torch.Tensor:
    """
    Executes `einops.rearrange(tensor, pattern, **axes_lengths)`.
    Falls back to explicit shape-verified PyTorch operations for common video/window patterns
    if the `einops` package is not installed in the current Python environment.
    """
    if _einops_rearrange is not None:
        return _einops_rearrange(tensor, pattern, **axes_lengths)

    norm_pat = " ".join(pattern.strip().split())
    if norm_pat == "b t c h w -> (b t) c h w":
        b, t, c, h, w = tensor.shape
        return tensor.reshape(b * t, c, h, w)
    if norm_pat == "(b t) c h w -> b t c h w":
        b = axes_lengths["b"]
        t = axes_lengths["t"]
        bt, c, h, w = tensor.shape
        assert bt == b * t, f"Shape mismatch: {bt} != {b} * {t}"
        return tensor.reshape(b, t, c, h, w)
    if norm_pat == "b w c h width -> b (w c) h width" or norm_pat == "b w c h w_sp -> b (w c) h w_sp":
        b, w_t, c, h, w_sp = tensor.shape
        return tensor.reshape(b, w_t * c, h, w_sp)
    if norm_pat == "b (h_head c_head) (nh wh) (nw ww) -> (b nh nw) h_head c_head (wh ww)":
        b, c, h, w = tensor.shape
        h_head = axes_lengths["h_head"]
        c_head = axes_lengths["c_head"]
        wh = axes_lengths["wh"]
        ww = axes_lengths["ww"]
        nh, nw = h // wh, w // ww
        x = tensor.reshape(b, h_head, c_head, nh, wh, nw, ww)
        x = x.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
        return x.reshape(b * nh * nw, h_head, c_head, wh * ww)
    if norm_pat == "(b nh nw) h_head c_head (wh ww) -> b (h_head c_head) (nh wh) (nw ww)":
        b = axes_lengths["b"]
        nh = axes_lengths["nh"]
        nw = axes_lengths["nw"]
        h_head = axes_lengths["h_head"]
        c_head = axes_lengths["c_head"]
        wh = axes_lengths["wh"]
        ww = axes_lengths["ww"]
        x = tensor.reshape(b, nh, nw, h_head, c_head, wh, ww)
        x = x.permute(0, 3, 4, 1, 5, 2, 6).contiguous()
        return x.reshape(b, h_head * c_head, nh * wh, nw * ww)
    raise NotImplementedError(f"Pattern '{pattern}' requires `einops` package.")
