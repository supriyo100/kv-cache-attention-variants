"""KV cache quantization in PyTorch: per-token, KIVI per-channel keys, TurboQuant rotation.

Asymmetric b-bit quantization of a group x: scale = (max-min)/(2^b-1), error <= scale/2,
so the range inside each group sets the error. Three ways to choose groups or reshape data:

* per-token groups (the naive layout): one scale per token per group of channels.
* KIVI (Liu et al., ICML 2024, arXiv:2402.02750; github.com/jy-yuan/KIVI): keys have
  a few large *outlier channels*, so keys are grouped per channel (across tokens) and
  values per token; the newest ``residual`` tokens stay in full precision.
* TurboQuant (Zandieh et al., arXiv:2504.19874; most-starred open implementation:
  github.com/0xSero/turboquant, GPL-3.0, linked only): multiply each vector by a random
  orthogonal matrix. The rotation spreads any outlier channel across all coordinates,
  which become nearly i.i.d. Gaussian with variance |x|^2/d. A fixed, data-oblivious
  Lloyd-Max codebook for N(0,1) then quantizes every coordinate near-optimally. Only the
  vector's norm is stored per token. (The paper adds a 1-bit QJL residual to make inner
  products unbiased; this module implements the MSE stage.)
"""

from __future__ import annotations

import functools
import math

import torch
from torch import Tensor


def quantize(x: Tensor, bits: int, dim: int) -> Tensor:
    """Fake-quantize with one (min, scale) per slice along ``dim``. Returns dequantized x."""
    lo = x.amin(dim=dim, keepdim=True)
    hi = x.amax(dim=dim, keepdim=True)
    levels = 2 ** bits - 1
    scale = torch.where(hi > lo, (hi - lo) / levels, torch.ones_like(hi))
    q = torch.clamp(torch.round((x - lo) / scale), 0, levels)
    return q * scale + lo


def quantize_grouped(x: Tensor, bits: int, mode: str, group: int = 32) -> Tensor:
    """x: (T, d). "per_token": groups of ``group`` channels within a token.
    "per_channel": groups of ``group`` tokens within a channel."""
    out = torch.empty_like(x)
    if mode == "per_token":
        for c0 in range(0, x.shape[1], group):
            out[:, c0:c0 + group] = quantize(x[:, c0:c0 + group], bits, dim=1)
    elif mode == "per_channel":
        for t0 in range(0, x.shape[0], group):
            out[t0:t0 + group] = quantize(x[t0:t0 + group], bits, dim=0)
    else:
        raise ValueError(mode)
    return out


def kivi(k: Tensor, v: Tensor, bits: int = 2, group: int = 32, residual: int = 32) -> tuple[Tensor, Tensor]:
    """KIVI layout: K per-channel, V per-token, last ``residual`` tokens kept exact."""
    cut = max(0, (k.shape[0] - residual) // group * group)
    k_hat, v_hat = k.clone(), v.clone()
    if cut:
        k_hat[:cut] = quantize_grouped(k[:cut], bits, "per_channel", group)
        v_hat[:cut] = quantize_grouped(v[:cut], bits, "per_token", group)
    return k_hat, v_hat


# --- TurboQuant (MSE stage) ----------------------------------------------------------------


@functools.cache
def lloyd_max_gaussian(bits: int, iters: int = 60, samples: int = 400_000) -> Tensor:
    """Optimal scalar codebook for N(0,1) with 2^bits levels (Lloyd's algorithm, seeded)."""
    g = torch.Generator().manual_seed(0)
    z = torch.randn(samples, generator=g)
    n = 2 ** bits
    c = torch.quantile(z, (torch.arange(n) + 0.5) / n)  # start at quantiles
    for _ in range(iters):
        edges = (c[1:] + c[:-1]) / 2
        idx = torch.bucketize(z, edges)
        c = torch.stack([z[idx == i].mean() for i in range(n)])
    return c


ROTATION_SEED = 0x7A11  # must be independent of the data being quantized


def random_rotation(d: int, seed: int = ROTATION_SEED) -> Tensor:
    """Haar-random orthogonal matrix. It must be drawn independently of the vectors it will
    rotate: a rotation correlated with the data does not make coordinates Gaussian."""
    g = torch.Generator().manual_seed(seed)
    qm, r = torch.linalg.qr(torch.randn(d, d, generator=g))
    return qm * torch.sign(torch.diagonal(r))  # uniform (Haar) orthogonal matrix


def turboquant(x: Tensor, bits: int, seed: int = ROTATION_SEED) -> Tensor:
    """Rotate each row, quantize coordinates with the N(0,1) Lloyd-Max codebook, rotate back.

    x: (T, d). Stored per row: ``bits`` per coordinate plus one norm.
    """
    d = x.shape[-1]
    rot = random_rotation(d, seed).to(x.dtype)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    y = (x / norm) @ rot.T * math.sqrt(d)  # coordinates ~ N(0, 1) after rotation
    code = lloyd_max_gaussian(bits).to(x.dtype)
    idx = torch.bucketize(y, (code[1:] + code[:-1]) / 2)
    return (code[idx] / math.sqrt(d)) @ rot * norm


# --- test data and metrics -----------------------------------------------------------------


def synthetic_kv(t: int = 512, d: int = 128, outlier_channels: int = 4, outlier_scale: float = 15.0,
                 seed: int = 0) -> tuple[Tensor, Tensor]:
    """Keys with persistent outlier channels (the pattern KIVI reports); Gaussian values."""
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(t, d, generator=g)
    ch = torch.randperm(d, generator=g)[:outlier_channels]
    sign = torch.sign(torch.randn(outlier_channels, generator=g))
    k[:, ch] = k[:, ch] * 2 + outlier_scale * sign
    v = torch.randn(t, d, generator=g)
    return k, v


def attention_out(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    return torch.nn.functional.scaled_dot_product_attention(q[None], k[None], v[None])[0]


def rel_error(a: Tensor, b: Tensor) -> float:
    return float((a - b).norm() / b.norm())


def bytes_per_elem(bits: int, group: int, meta_bits: int = 16) -> float:
    """Group quantization storage: bits plus one fp16 scale and one fp16 zero per group."""
    return (bits + 2 * meta_bits / group) / 8


def turboquant_bytes_per_elem(bits: int, d: int, meta_bits: int = 16) -> float:
    """TurboQuant storage: bits plus one fp16 norm per d-dimensional vector."""
    return (bits + meta_bits / d) / 8
