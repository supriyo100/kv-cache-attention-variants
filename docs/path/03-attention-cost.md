---
title: "What attention costs: FLOPs, bytes and the T×T matrix"
description: >-
  The cost of scaled dot-product attention in LLM inference: tensor shapes, FLOPs per layer,
  the quadratic score matrix, when attention overtakes the linear layers, and the arithmetic
  intensity of decode attention for MHA, GQA and MLA.
step: 3
layer: kernel
---

# 3. What attention costs

<p class="lede">Attention is the only part of a transformer where tokens look at each other,
and the only part whose cost grows with context length. This step writes out its shapes and
counts its FLOPs and bytes, so the next eight steps have numbers to work with.</p>

## Problem

The linear layers cost the same per token at position 10 and at position 100,000. Attention
does not. Where does attention's cost come from, and at what context length does it take over?

## The computation

For one head with $T$ queries and $T$ keys:

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right) V,
\qquad Q, K, V \in \mathbb{R}^{T \times d_h},
$$

where $M$ is the causal mask ($-\infty$ above the diagonal). In three stages:

$$
S = \frac{QK^\top}{\sqrt{d_h}} \in \mathbb{R}^{T\times T},\qquad
P_{ij} = \frac{e^{S_{ij} - m_i}}{\sum_{j'} e^{S_{ij'} - m_i}},\quad m_i = \max_j S_{ij},\qquad
O = PV \in \mathbb{R}^{T\times d_h}.
$$

Subtracting the row max $m_i$ changes nothing mathematically and keeps $e^{x}$ from
overflowing in FP16. It also means **every row needs its maximum before it can be
normalized**, which is the dependency FlashAttention removes in step [10](10-flashattention.md).

<figure class="ie-fig">
<svg viewBox="0 0 760 230" role="img" aria-label="Tensor shapes: Q is T by d, K transpose is d by T, their product S is T by T, softmax gives P of T by T, times V gives O of T by d">
<rect x="20" y="40" width="40" height="150" class="compute"/><text x="40" y="208" text-anchor="middle" class="t-mono">Q  T×d</text>
<text x="75" y="120" text-anchor="middle">×</text>
<rect x="90" y="40" width="150" height="40" class="memory"/><text x="165" y="98" text-anchor="middle" class="t-mono">Kᵀ  d×T</text>
<text x="258" y="120" text-anchor="middle">=</text>
<rect x="275" y="40" width="150" height="150" class="waste"/><text x="350" y="120" text-anchor="middle" class="t-b">S  T×T</text>
<path d="M275,40 L425,190" class="ln-d"/><text x="350" y="208" text-anchor="middle" class="t-sm">causal: upper triangle masked</text>
<text x="450" y="120" text-anchor="middle" class="t-sm">softmax</text>
<rect x="475" y="40" width="150" height="150" class="waste"/><text x="550" y="120" text-anchor="middle" class="t-b">P  T×T</text>
<text x="640" y="120" text-anchor="middle">×</text>
<rect x="655" y="40" width="40" height="150" class="memory"/><text x="675" y="208" text-anchor="middle" class="t-mono">V  T×d</text>
<text x="715" y="120" text-anchor="middle">=</text>
<rect x="728" y="40" width="22" height="150" class="compute"/><text x="739" y="30" text-anchor="middle" class="t-mono">O</text>
</svg>
<figcaption><b>Only S and P grow as T².</b> Inputs and output are $T \times d_h$, linear in
context. The two intermediate matrices are $T \times T$. Materialize them in HBM and the
memory and traffic go quadratic. That is a kernel problem (step 10), not a KV-cache problem.</figcaption>
</figure>

## FLOPs

Per layer, summed over $H_q$ query heads, ignoring the mask:

$$
F_{\text{attn}} = \underbrace{2T^2 d_h}_{QK^\top} + \underbrace{2T^2 d_h}_{PV} = 4 H_q d_h T^2 \quad\text{(halved by causality in prefill)}.
$$

The linear layers cost $2P/L$ per token per layer. Per token, attention over $T$ previous
tokens costs $4 H_q d_h T$ per layer. Attention overtakes the matmuls when

$$
4 L H_q d_h T > 2P \;\Longrightarrow\; T > T_\times = \frac{P}{2 L H_q d_h}.
$$

For Llama-3-8B, $T_\times = 8\times10^9 / (2 \cdot 32 \cdot 32 \cdot 128) \approx 30{,}500$
tokens. Below about 30K tokens of context, attention is a minority of decode FLOPs. Above it,
attention dominates. (Causal prefill averages $T/2$ of context, so the prefill crossover is
about 61K.)

## Bytes, and why decode attention is a memory problem

During decode a sequence adds one query and reads its whole KV cache:

$$
Q_{\text{attn}} = T \cdot m_{\text{tok}} = T \cdot L \cdot 2 H_{kv} d_h s \text{ bytes}.
$$

KV reads overtake weight reads when $B T m_{\text{tok}} > sP$, at context
$T > sP / (B\, m_{\text{tok}})$. For Llama-3-8B ($m_{\text{tok}} = 128$ KiB): 122K tokens at
$B=1$, but only **1,900 tokens at $B=64$**. Under real serving loads, the cache, not the
weights, is the dominant memory traffic. That is why steps 4–9 exist.

### The arithmetic intensity of the attention kernel itself

Inside the decode attention kernel, per cached token per layer:

| Variant | FLOPs | Bytes read | Intensity (FLOP/byte, BF16) |
|---|---|---|---|
| MHA ($H_{kv} = H_q$) | $4 H_q d_h$ | $2 H_q d_h s$ | $2/s = 1$ |
| GQA, group size $g = H_q / H_{kv}$ | $4 H_q d_h$ | $2 H_{kv} d_h s$ | $g$ (4 for Llama-3) |
| MQA ($H_{kv} = 1$) | $4 H_q d_h$ | $2 d_h s$ | $H_q$ (32) |
| MLA, absorbed (DeepSeek-V3) | $H_q (2 d_c + 2 d_R + 2 d_c)$ | $(d_c + d_R) s$ | $\approx 242$ |

The MLA row is computed for $H_q = 128$, $d_c = 512$, $d_R = 64$: 278,528 FLOPs over 1,152
bytes. Grouping is not only a capacity saving. Each byte of K/V fetched now serves $g$ query
heads, so intensity rises by $g$. MLA in absorbed form reaches about 242 FLOP/byte, close to
the H100 ridge, which is why DeepSeek's FlashMLA kernel is tuned as a *compute*-bound kernel.
Step [5](05-attention-variants.md) derives the absorption.

## Implementation

- `inference_lab.flash.naive_attention` is the three-stage computation above, materializing
  $S$ and $P$.
- `inference_lab.flash.tiled_attention` gives the same result without ever forming $S$
  (step 10). The test `test_tiled_matches_naive` checks equality to $10^{-10}$.

## Trade-offs and failure modes

- **Quadratic memory, not just compute.** One 32K × 32K FP16 score matrix is 2 GiB *per
  head*. A naive kernel that materializes all 32 heads of a layer at once needs 64 GiB. Long
  context was impossible before IO-aware kernels for this reason, not for lack of FLOPs.
- **Softmax precision.** Scores must be accumulated in FP32 even when Q and K are FP16/FP8, or
  the row max and the denominator lose precision at long context.
- **Masking costs.** Causal masking halves useful FLOPs, but a kernel that does not skip fully
  masked tiles still pays for them.

## Questions

1. For Llama-3-70B ($L=80$, $H_q=64$, $d_h=128$, $P = 70.6\text{B}$), at what context does
   attention FLOPs equal linear FLOPs?
2. Why does GQA raise the arithmetic intensity of decode attention but not of prefill
   attention?
3. If you shard heads across 8 GPUs with tensor parallelism, what happens to each GPU's
   share of the $T \times T$ problem?

<div class="ie-prov" markdown>

**Provenance**

Source
: Scaled dot-product attention (Vaswani et al., 2017). MQA (Shazeer, 2019), GQA (Ainslie et
  al., 2023), MLA (DeepSeek-V2, 2024). FlashMLA (github.com/deepseek-ai/FlashMLA) for the
  compute-bound decode kernel.

Derivation
: The crossover lengths and the per-variant intensity table are derived on this page.

Implementation
: `inference_lab.flash` (naive and tiled attention; tested).

Experiment
: None. All numbers are arithmetic on model shapes.

</div>
