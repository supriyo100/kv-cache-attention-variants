"""MLA weight absorption, built on the course's MultiHeadLatentAttention module.

``kv_cache_variants.attention.mla.MultiHeadLatentAttention`` (the upstream course module)
caches the latent c_t = W_DKV x_t and the shared RoPE key, and at every step
*up-projects the whole cache* into per-head K and V (``k_up_proj(latent)``,
``v_up_proj(latent)``). That is the explicit form: correct, and simple, but each decode
step costs O(T * H * d_h * d_c) FLOPs and materialises K and V for all T cached tokens.

DeepSeek-V2 (arXiv:2405.04434) avoids that with associativity. Per head i:

    q_i^T (W_UK^i c_t) = ((W_UK^i)^T q_i)^T c_t      -> absorb W_UK into the query
    sum_t p_t (W_UV^i c_t) = W_UV^i (sum_t p_t c_t)   -> apply W_UV once, after attention
    W_O^i W_UV^i                                      -> can be pre-multiplied (fold_output)

``absorbed_forward`` computes exactly what ``MultiHeadLatentAttention.forward`` computes,
from the same module's weights, without building K or V. FlashMLA
(github.com/deepseek-ai/FlashMLA) is the production GPU kernel for this absorbed decode.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from kv_cache_variants.attention.mla import MLACache, MultiHeadLatentAttention
from kv_cache_variants.rope import apply_rotary


def up_projections(mla: MultiHeadLatentAttention) -> tuple[Tensor, Tensor]:
    """Per-head W_UK, W_UV as (H, d_h, d_c) views of the module's nn.Linear weights."""
    h, dh, dc = mla.num_heads, mla.head_dim, mla.latent_dim
    return mla.k_up_proj.weight.view(h, dh, dc), mla.v_up_proj.weight.view(h, dh, dc)


def folded_output(mla: MultiHeadLatentAttention) -> Tensor:
    """W_O^i W_UV^i for every head: (H, d_model, d_c). Lets the output skip d_h entirely."""
    h, dh = mla.num_heads, mla.head_dim
    _, w_uv = up_projections(mla)
    w_o = mla.out_proj.weight.view(-1, h, dh).permute(1, 0, 2)  # (H, d_model, d_h)
    return w_o @ w_uv


def absorbed_forward(mla: MultiHeadLatentAttention, x: Tensor, past_kv: MLACache | None = None,
                     rope: tuple[Tensor, Tensor] | None = None,
                     fold_output: bool = False) -> tuple[Tensor, MLACache]:
    """Same inputs and outputs as ``mla(x, past_kv, rope)``, computed in latent space.

    x: (batch, seq, d_model). Returns (output, (latent, k_rope)).
    """
    batch, seq, _ = x.shape
    h, dh, dr = mla.num_heads, mla.head_dim, mla.rope_head_dim
    offset = 0 if past_kv is None else past_kv[0].shape[1]

    latent = mla.kv_down_proj(x)  # (B, s, d_c)   what gets cached
    k_rope = mla.k_rope_proj(x)  # (B, s, d_R)   shared by all heads, cached
    if past_kv is not None:
        latent = torch.cat([past_kv[0], latent], dim=1)
        k_rope = torch.cat([past_kv[1], k_rope], dim=1)
    total = latent.shape[1]

    q_c = mla._split(mla.q_proj(x), h, dh)  # (B, H, s, d_h)
    q_r = mla._split(mla.q_rope_proj(x), h, dr)  # (B, H, s, d_R)
    k_r = k_rope.unsqueeze(1)  # (B, 1, T, d_R)
    if rope is not None:
        cos, sin = rope
        q_r = apply_rotary(q_r, cos, sin, offset=offset)
        k_r = apply_rotary(k_r, cos, sin, offset=0)

    w_uk, w_uv = up_projections(mla)
    q_lat = torch.einsum("bhsd,hdc->bhsc", q_c, w_uk)  # W_UK absorbed: (B, H, s, d_c)
    lat = latent.unsqueeze(1)  # (B, 1, T, d_c): ONE shared "KV head"
    scores = (q_lat @ lat.transpose(-1, -2) + q_r @ k_r.transpose(-1, -2)) / math.sqrt(dh + dr)
    if past_kv is None and seq > 1:  # causal prefill, matching the module's is_causal=True
        mask = torch.ones(seq, total, dtype=torch.bool, device=x.device).triu(1 + total - seq)
        scores = scores.masked_fill(mask, float("-inf"))
    p = scores.softmax(dim=-1)
    o_lat = p @ lat  # (B, H, s, d_c): attend over latents

    if fold_output:  # W_O W_UV pre-multiplied: never form the per-head d_h output
        return torch.einsum("bhsc,hmc->bsm", o_lat, folded_output(mla)), (latent, k_rope)
    o = torch.einsum("bhsc,hdc->bhsd", o_lat, w_uv)  # W_UV once per head
    return mla.out_proj(o.transpose(1, 2).reshape(batch, seq, h * dh)), (latent, k_rope)


def cache_elems_per_token(d_c: int, d_r: int) -> int:
    return d_c + d_r


def equivalent_mha_elems_per_token(heads: int, d_h: int) -> int:
    return 2 * heads * d_h


def decode_flops(mla: MultiHeadLatentAttention, context: int, absorbed: bool) -> int:
    """Attention-side FLOPs of one decode step (one new token), excluding shared projections.

    explicit: up-project T latents to K and V for every head, then score and sum.
    absorbed: project the one query into latent space, score and sum over latents.
    """
    h, dh, dc, dr = mla.num_heads, mla.head_dim, mla.latent_dim, mla.rope_head_dim
    if absorbed:
        return 2 * h * dh * dc * 2 + h * context * (2 * dc + 2 * dr + 2 * dc)
    return 2 * context * dc * h * dh * 2 + h * context * (2 * dh + 2 * dr + 2 * dh)
