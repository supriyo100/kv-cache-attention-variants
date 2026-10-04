"""Speculative decoding in PyTorch: the acceptance rule, its guarantee, and its payoff.

A draft proposes gamma tokens; the target scores them in one forward pass. Draft token
x ~ q is accepted with probability min(1, p(x)/q(x)); on the first rejection a
replacement is sampled from norm(max(0, p - q)). The output is distributed exactly as p
(Leviathan et al., arXiv:2211.17192; Chen et al., arXiv:2302.01318).

With i.i.d. acceptance alpha: E[tokens per target pass] = (1 - alpha^(gamma+1)) / (1 - alpha),
and speedup = E / (gamma * c + 1) for draft cost c relative to one target pass.

Recent drafters (most-starred training/eval codebase: github.com/deepseek-ai/DeepSpec,
covering EAGLE-3, DFlash and DSpark) draft a whole block in one pass. Acceptance then
decays with position, and verifying low-confidence tail tokens wastes batch capacity
at high concurrency. DSpark (arXiv:2607.05147) schedules verification length by
confidence; ``confidence_schedule`` models that choice.
"""

from __future__ import annotations

import torch
from torch import Tensor


def acceptance_rate(p: Tensor, q: Tensor) -> float:
    """alpha = sum_x min(p, q) = 1 - TV(p, q)."""
    return float(torch.minimum(p, q).sum())


def speculative_step(p: Tensor, q: Tensor, gen: torch.Generator) -> tuple[int, bool]:
    """One draft position. Returns (emitted token, accepted?). Emitted ~ p exactly."""
    x = int(torch.multinomial(q, 1, generator=gen))
    if float(torch.rand((), generator=gen)) < min(1.0, float(p[x] / q[x])):
        return x, True
    resid = (p - q).clamp_min(0)
    return int(torch.multinomial(resid / resid.sum(), 1, generator=gen)), False


def expected_tokens(alpha: float, gamma: int) -> float:
    if alpha >= 1.0:
        return gamma + 1.0
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def speedup(alpha: float, gamma: int, draft_cost: float) -> float:
    return expected_tokens(alpha, gamma) / (gamma * draft_cost + 1)


def best_gamma(alpha: float, draft_cost: float, max_gamma: int = 16) -> tuple[int, float]:
    return max(((g, speedup(alpha, g, draft_cost)) for g in range(1, max_gamma + 1)), key=lambda x: x[1])


def simulate_chain(alpha: float, gamma: int, steps: int, seed: int = 0) -> float:
    """Monte-Carlo mean tokens per target pass with i.i.d. acceptance alpha."""
    gen = torch.Generator().manual_seed(seed)
    accept = torch.rand(steps, gamma, generator=gen) < alpha
    n_accepted = accept.int().cumprod(dim=1).sum(dim=1)  # length of the accepted prefix
    return float((n_accepted + 1).float().mean())


# --- position-dependent acceptance (block drafters) ---------------------------------------


def expected_tokens_positional(alphas: Tensor) -> Tensor:
    """E[tokens] when verifying the first k drafts, for k = 0..len(alphas).

    Draft i is accepted only if drafts 1..i are: E_k = 1 + sum_{j<=k} prod_{i<=j} alpha_i.
    """
    return torch.cat([torch.ones(1), 1 + alphas.cumprod(0).cumsum(0)])


def confidence_schedule(alphas: Tensor, slot_budget: int) -> tuple[Tensor, float]:
    """Choose how many drafts to verify per request under a shared slot budget.

    alphas: (N, gamma) per-position acceptance probabilities of N requests' drafted blocks.
    Every request always gets one slot (the target's own next token); verifying draft j of
    request r costs one more slot and yields prod_{i<=j} alpha_{r,i} expected tokens. Those
    marginal values decrease along each draft, so taking the largest ones across all
    requests (greedy) is optimal and always selects prefixes: verify a draft iff its
    cumulative acceptance probability clears the threshold the budget implies.

    Returns (verify length per request, total expected tokens).
    """
    n, gamma = alphas.shape
    extra = slot_budget - n
    if extra < 0:
        raise ValueError("budget smaller than one slot per request")
    value = alphas.cumprod(dim=1)  # (N, gamma) marginal expected tokens
    take = torch.zeros_like(value, dtype=torch.bool)
    if extra > 0:
        flat = value.flatten().topk(min(extra, value.numel())).indices
        take.view(-1)[flat] = True
    lengths = take.int().cumprod(dim=1).sum(dim=1)  # prefixes (ties at equal value stay prefixes)
    total = n + float((value * (torch.arange(gamma) < lengths[:, None])).sum())
    return lengths, total


def fixed_schedule(alphas: Tensor, slot_budget: int) -> tuple[int, float]:
    """Baseline: the same verify length k for every request, as large as the budget allows."""
    n, gamma = alphas.shape
    k = min(gamma, slot_budget // n - 1)
    return k, n + float(alphas.cumprod(dim=1)[:, :k].sum())
