"""KV eviction in PyTorch: keep a budget of tokens, drop the rest.

Three published policies reduced to their selection rule:

* StreamingLLM (Xiao et al., ICLR 2024, arXiv:2309.17453; github.com/mit-han-lab/streaming-llm,
  the most-starred eviction implementation): keep the first ``sinks`` tokens ("attention
  sinks") plus a recent window. Content-blind.
* H2O (Zhang et al., NeurIPS 2023, arXiv:2306.14048): recent window plus "heavy hitters",
  the tokens with the largest accumulated attention over past queries.
* SnapKV (Li et al., arXiv:2404.14469; github.com/FasterDecoding/SnapKV): the last ``obs``
  prompt queries vote; votes are max-pooled over neighbours so clusters survive; keep
  the top tokens plus the observation window. NVIDIA/kvpress and KVCache-Factory package
  these and many more policies for Hugging Face models.

Test bed: a context in which a few "needle" tokens carry what later queries look for.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def streaming(t: int, budget: int, sinks: int = 4) -> Tensor:
    if budget >= t:
        return torch.arange(t)
    return torch.cat([torch.arange(sinks), torch.arange(t - (budget - sinks), t)])


def h2o(attn_hist: Tensor, budget: int, recent: int) -> Tensor:
    """attn_hist: (n_queries, T) attention probabilities observed so far."""
    t = attn_hist.shape[1]
    if budget >= t:
        return torch.arange(t)
    acc = attn_hist.sum(dim=0)
    acc[t - recent:] = float("-inf")
    heavy = acc.topk(budget - recent).indices
    return torch.cat([heavy, torch.arange(t - recent, t)]).sort().values


def snapkv(q_obs: Tensor, k: Tensor, budget: int, kernel: int = 7) -> Tensor:
    """q_obs: (obs, d), the last prompt queries; k: (T, d), all prompt keys."""
    t, obs = k.shape[0], q_obs.shape[0]
    if budget >= t:
        return torch.arange(t)
    votes = (q_obs @ k[: t - obs].T / math.sqrt(k.shape[1])).softmax(dim=-1).sum(dim=0)
    pooled = torch.nn.functional.max_pool1d(votes[None, None], kernel, stride=1, padding=kernel // 2)[0, 0]
    top = pooled.topk(budget - obs).indices
    return torch.cat([top, torch.arange(t - obs, t)]).sort().values


def needle_context(t: int = 1024, d: int = 64, needles: int = 8, strength: float = 16.0, seed: int = 0):
    """Keys/values where ``needles`` positions align with a hidden query direction.

    ``strength`` sets how peaked attention is. With the default, needles hold most of the
    softmax mass (as in retrieval heads); with small values attention is diffuse and no
    small budget preserves the output.
    """
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(t, d, generator=g)
    v = torch.randn(t, d, generator=g)
    direction = torch.randn(d, generator=g)
    direction /= direction.norm()
    pos = (16 + torch.randperm(t - 144, generator=g)[:needles]).sort().values
    k[pos] += strength * direction
    k[:4] += 3.0 * direction  # early tokens act as sinks
    return k, v, direction, pos


def queries_toward(direction: Tensor, n: int, scale: float = 4.0, noise: float = 0.5, seed: int = 1) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    return scale * direction + noise * torch.randn(n, direction.shape[0], generator=g)


def attention_probs(q: Tensor, k: Tensor) -> Tensor:
    return (q @ k.T / math.sqrt(k.shape[1])).softmax(dim=-1)


def output_error(q: Tensor, k: Tensor, v: Tensor, keep: Tensor) -> float:
    full = attention_probs(q, k) @ v
    part = attention_probs(q, k[keep]) @ v[keep]
    return float((part - full).norm() / full.norm())
