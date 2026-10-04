"""IO-aware attention in PyTorch: tiling, online softmax, split-KV merging.

Reference semantics of FlashAttention (Dao et al., 2022, arXiv:2205.14135;
github.com/Dao-AILab/flash-attention), written as readable PyTorch loops rather than a
fused kernel. Tensors use the course convention (batch, heads, seq, head_dim); every
function also accepts any leading batch dimensions. Results are checked against
``torch.nn.functional.scaled_dot_product_attention`` in the tests.

* ``naive_attention`` materialises S = QK^T/sqrt(d) and P = softmax(S), as three
  separate kernels would.
* ``tiled_attention`` streams K/V tiles and keeps a running max m, denominator l and
  unnormalised output per query row, rescaling by exp(m_old - m_new) when the max
  rises. Exact, never forms more than one (block_q x block_k) score tile.
* ``merge_states`` combines attention over disjoint KV sets with their log-sum-exp:
  the identity behind FlashDecoding (split-KV), FlashInfer's cascade attention
  (``flashinfer/cascade.py``) and context parallelism.
* ``io_naive`` / ``io_flash`` count HBM element traffic (FlashAttention Theorem 2).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _causal_mask(n_q: int, n_k: int, q0: int = 0, k0: int = 0, device=None) -> Tensor:
    """True where key position > query position; queries are the last n_q of n_k overall."""
    qpos = torch.arange(q0, q0 + n_q, device=device)[:, None]
    kpos = torch.arange(k0, k0 + n_k, device=device)[None, :]
    return kpos > qpos


def naive_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = False) -> Tensor:
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])  # (..., T_q, T_k): materialised
    if causal:
        t_q, t_k = s.shape[-2:]
        s = s.masked_fill(_causal_mask(t_q, t_k, q0=t_k - t_q, device=q.device), float("-inf"))
    return s.softmax(dim=-1) @ v


def tiled_attention(q: Tensor, k: Tensor, v: Tensor, block_q: int = 64, block_k: int = 64,
                    causal: bool = False) -> tuple[Tensor, Tensor]:
    """FlashAttention forward. Returns (output, lse) with lse shaped (..., T_q)."""
    *lead, t_q, d = q.shape
    t_k = k.shape[-2]
    scale = 1.0 / math.sqrt(d)
    offset = t_k - t_q  # causal alignment: queries are the last t_q positions
    acc_t = torch.promote_types(q.dtype, torch.float32)  # FP32 accumulation for FP16/BF16 inputs
    out = torch.empty_like(q, dtype=acc_t)
    lse = torch.empty(*lead, t_q, dtype=acc_t, device=q.device)
    for qs in range(0, t_q, block_q):
        qi = q[..., qs:qs + block_q, :].to(acc_t)
        rows = qi.shape[-2]
        m = torch.full((*lead, rows), float("-inf"), dtype=acc_t, device=q.device)
        l_ = torch.zeros(*lead, rows, dtype=acc_t, device=q.device)
        acc = torch.zeros(*lead, rows, d, dtype=acc_t, device=q.device)
        for ks in range(0, t_k, block_k):
            if causal and ks > qs + rows - 1 + offset:
                break  # this tile and all later ones are entirely in the future
            kj = k[..., ks:ks + block_k, :].to(acc_t)
            vj = v[..., ks:ks + block_k, :].to(acc_t)
            s = qi @ kj.transpose(-1, -2) * scale
            if causal:
                s = s.masked_fill(_causal_mask(rows, kj.shape[-2], qs + offset, ks, q.device), float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1))
            alpha = torch.exp(m - m_new)  # rescale what was accumulated under the old max
            p = torch.exp(s - m_new[..., None])
            l_ = alpha * l_ + p.sum(dim=-1)
            acc = alpha[..., None] * acc + p @ vj
            m = m_new
        out[..., qs:qs + rows, :] = acc / l_[..., None]
        lse[..., qs:qs + rows] = m + torch.log(l_)
    return out.to(q.dtype), lse


def partial_state(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Attention of q over one KV chunk, as (normalised output, lse)."""
    return tiled_attention(q, k, v, block_q=q.shape[-2], block_k=k.shape[-2])


def merge_states(o_a: Tensor, lse_a: Tensor, o_b: Tensor, lse_b: Tensor) -> tuple[Tensor, Tensor]:
    """Exact merge of attention over disjoint KV sets A and B.

    lse = log(e^lse_a + e^lse_b);  o = e^(lse_a - lse) o_a + e^(lse_b - lse) o_b
    """
    lse = torch.logaddexp(lse_a, lse_b)
    return torch.exp(lse_a - lse)[..., None] * o_a + torch.exp(lse_b - lse)[..., None] * o_b, lse


def split_kv_attention(q: Tensor, k: Tensor, v: Tensor, splits: int) -> Tensor:
    """FlashDecoding semantics: attend over ``splits`` KV chunks independently, then merge."""
    ks, vs = torch.tensor_split(k, splits, dim=-2), torch.tensor_split(v, splits, dim=-2)
    o, lse = partial_state(q, ks[0], vs[0])
    for kc, vc in zip(ks[1:], vs[1:]):
        o, lse = merge_states(o, lse, *partial_state(q, kc, vc))
    return o


# --- HBM traffic model (elements; multiply by bytes per element) ---------------------------


def io_naive(n: int, d: int) -> int:
    """Read Q,K (2Nd); write S, read S, write P, read P (4N^2); read V, write O (2Nd)."""
    return 4 * n * d + 4 * n * n


def io_flash(n: int, d: int, sram_elems: int) -> int:
    """Q and O once; K and V streamed once per query tile of B_r ~ M/(4d) rows."""
    b_r = max(1, min(n, sram_elems // (4 * d)))
    t_r = -(-n // b_r)
    return 2 * n * d + t_r * 2 * n * d
