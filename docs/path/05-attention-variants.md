---
title: "MHA vs MQA vs GQA vs MLA: KV cache size, bandwidth and the MLA absorption trick"
description: >-
  Multi-head, multi-query, grouped-query and multi-head latent attention compared by KV bytes per
  token, decode arithmetic intensity and quality, with a full derivation of how MLA absorbs its
  up-projections so full keys and values are never materialized, and why RoPE needs a decoupled key.
step: 5
layer: model architecture
---

# 5. MHA, MQA, GQA, MLA

<p class="lede">The cache equation has one term the model designer controls directly: what each
token leaves behind per layer. Four designs answer that differently. Three of them change how
many K/V heads exist. The fourth, MLA, changes what is stored at all, and its key step is an
algebraic rearrangement, not a smaller head count.</p>

## Problem

With multi-head attention, Llama-2-7B leaves 512 KiB per token. A 32K context is 16 GiB, per
sequence. How do you cut $m_{\text{tok}}$ without destroying quality?

## The naive solutions, and why they fall short

- **Fewer or smaller heads overall.** This shrinks the cache, but also the model's attention
  capacity: queries lose expressiveness along with keys.
- **Fewer layers.** Each layer is model capacity. This is a different, smaller model.
- **Compress the cache after the fact (quantize or evict).** Valid and complementary
  (step 11), but it doesn't change what the architecture asks you to store.

The insight behind all four designs: **queries are not cached, keys and values are.** Keep
many query heads and make keys and values cheaper.

## MHA, MQA, GQA: sharing K/V heads

With $H_q$ query heads and $H_{kv}$ key/value heads, query head $i$ reads K/V head
$\lfloor i / g \rfloor$, where $g = H_q / H_{kv}$ is the group size:

$$
o_i = \mathrm{softmax}\!\left(\frac{q_i K_{\lfloor i/g\rfloor}^\top}{\sqrt{d_h}}\right) V_{\lfloor i/g\rfloor},
\qquad m_{\text{tok}} = 2 L H_{kv} d_h s.
$$

| | MHA | GQA | MQA |
|---|---|---|---|
| $H_{kv}$ | $H_q$ | $H_q/g$ (often 8) | 1 |
| K/V per token per layer | $2H_q d_h$ | $2 H_q d_h / g$ | $2 d_h$ |
| Llama-3-8B shape, BF16 | 512 KiB | 128 KiB ($g=4$) | 16 KiB |
| Decode attention intensity | 1 | $g$ | $H_q$ |
| Quality (published) | baseline | close to MHA at $g \le 8$ | measurable loss, less stable training |
| Tensor parallel across $N$ GPUs | heads split cleanly | KV replicated if $N > H_{kv}$ | KV replicated on every GPU |

GQA was introduced as a way to *uptrain* existing MHA checkpoints with about 5% of the original
pre-training compute, and it became the default for open dense models (Llama 2 70B onward,
Llama 3, Mistral, Qwen 2+). MQA appeared earlier (PaLM, Falcon) but its quality and training
stability cost pushed most later models to GQA.

## MLA: change what is cached

Multi-head Latent Attention (DeepSeek-V2, kept in V3 and R1) caches neither per-head keys nor
per-head values. For hidden state $h_t \in \mathbb{R}^d$ it stores one latent vector:

$$
c_t = W_{DKV}\, h_t \in \mathbb{R}^{d_c}, \qquad d_c = 512 \ll 2 H d_h = 32{,}768.
$$

Per head $i$, keys and values are linear up-projections of that latent:

$$
k^{C}_{t,i} = W_{UK}^{i} c_t, \qquad v_{t,i} = W_{UV}^{i} c_t, \qquad W_{UK}^{i}, W_{UV}^{i} \in \mathbb{R}^{d_h \times d_c}.
$$

If you computed $k$ and $v$ from $c_t$ at every step, you would save memory but pay
$O(T \cdot H \cdot d_h \cdot d_c)$ extra FLOPs per step and materialize full K/V anyway. MLA
avoids both by **absorbing** the up-projections.

### Absorbing $W_{UK}$ into the query

The content score of query head $i$ against cached token $t$ is

$$
\big(q^{C}_i\big)^{\!\top} k^{C}_{t,i}
= \big(q^{C}_i\big)^{\!\top} W_{UK}^{i}\, c_t
= \Big(\underbrace{(W_{UK}^{i})^{\top} q^{C}_i}_{\tilde q_i \,\in\, \mathbb{R}^{d_c}}\Big)^{\!\top} c_t .
$$

Matrix multiplication is associative, so we can project the *one* query into latent space
instead of projecting *every* cached latent into key space. Per step that costs one
$d_c \times d_h$ product per head, independent of $T$. All heads then score against the same
cached $c_t$, so in the absorbed form MLA decode is attention with **a single shared KV head of
width $d_c$** (MQA-like memory traffic). Each head still has its own $\tilde q_i$, so heads
keep distinct attention patterns.

### Absorbing $W_{UV}$ into the output

$$
o_i = \sum_t p_{t,i}\, v_{t,i} = \sum_t p_{t,i}\, W_{UV}^{i} c_t = W_{UV}^{i} \Big(\underbrace{\textstyle\sum_t p_{t,i}\, c_t}_{\bar c_i \,\in\, \mathbb{R}^{d_c}}\Big).
$$

Attend over the latents first, then up-project once per head. The output projection
$W_O = [W_O^1 \cdots W_O^H]$ follows, so $W_O^i W_{UV}^i$ can be pre-multiplied into one
matrix. **Full K and V are never materialized during decode.**

### Why RoPE needs a decoupled key

RoPE rotates keys by a position-dependent matrix $R_t$. With rotated keys the score becomes
$q_i^\top R_s^\top R_t W_{UK}^{i} c_t$. The rotation $R_{t-s}$ sits *between* the query and
$W_{UK}^{i}$ and depends on both positions, so $W_{UK}^{i}$ can no longer be folded into a
position-independent query. MLA's fix is to carry position in a small separate channel:

$$
k^{R}_t = \mathrm{RoPE}_t\big(W_{KR} h_t\big) \in \mathbb{R}^{d_R}\ (\text{shared by all heads}),\qquad
\text{score}_{t,i} = \frac{\tilde q_i^{\top} c_t + (q^{R}_i)^{\top} k^{R}_t}{\sqrt{d_h + d_R}}.
$$

The cache per token per layer is therefore $c_t$ plus $k^R_t$: $d_c + d_R = 512 + 64 = 576$
elements. There's no factor of 2, because no separate V is stored.

<figure class="ie-fig">
<svg viewBox="0 0 760 300" role="img" aria-label="MLA dataflow: hidden state is down-projected to a 512-wide latent and a 64-wide rope key, both cached; at decode the query is projected into latent space, scored against the cached latents, the attended latent is up-projected by W_UV and W_O">
<rect x="20" y="120" width="90" height="40" class="box"/><text x="65" y="145" text-anchor="middle" class="t-b">h_t (7168)</text>
<path d="M110,140 L170,95" class="ln"/><path d="M110,140 L170,190" class="ln"/>
<rect x="170" y="75" width="110" height="40" class="compute"/><text x="225" y="100" text-anchor="middle" class="t-sm">W_DKV ↓</text>
<rect x="170" y="170" width="110" height="40" class="compute"/><text x="225" y="195" text-anchor="middle" class="t-sm">W_KR, RoPE</text>
<path d="M280,95 L330,95" class="ln"/><path d="M280,190 L330,190" class="ln"/>
<rect x="330" y="60" width="150" height="160" class="memory" style="opacity:.25"/>
<text x="405" y="52" text-anchor="middle" class="t-b" style="fill:var(--ie-memory)">KV cache (per token, per layer)</text>
<rect x="340" y="75" width="130" height="40" class="memory"/><text x="405" y="100" text-anchor="middle" class="t-b">c_t (512)</text>
<rect x="340" y="170" width="130" height="40" class="memory"/><text x="405" y="195" text-anchor="middle" class="t-b">k^R_t (64)</text>
<text x="405" y="144" text-anchor="middle" class="t-sm">576 elements</text>
<text x="405" y="160" text-anchor="middle" class="t-sm">vs 32,768 for MHA</text>
<rect x="540" y="20" width="200" height="36" class="box"/><text x="640" y="43" text-anchor="middle" class="t-sm">query head i: q_i, q^R_i</text>
<path d="M640,56 L640,75" class="ln"/>
<rect x="540" y="75" width="200" height="40" class="compute"/><text x="640" y="92" text-anchor="middle" class="t-sm">q̃_i = W_UKᵢᵀ q_i  (absorbed)</text><text x="640" y="107" text-anchor="middle" class="t-sm">width 512</text>
<path d="M540,95 L470,95" class="ln"/><path d="M540,190 L470,190" class="ln"/>
<rect x="540" y="130" width="200" height="40" class="compute"/><text x="640" y="155" text-anchor="middle" class="t-sm">scores: q̃_iᵀc_t + q^R_iᵀk^R_t → softmax</text>
<path d="M640,115 L640,130" class="ln"/>
<rect x="540" y="185" width="200" height="40" class="compute"/><text x="640" y="210" text-anchor="middle" class="t-sm">c̄_i = Σ p_t c_t   (attend latents)</text>
<path d="M640,170 L640,185" class="ln"/>
<rect x="540" y="240" width="200" height="40" class="compute"/><text x="640" y="265" text-anchor="middle" class="t-sm">W_O W_UVᵢ c̄_i  (absorbed, once)</text>
<path d="M640,225 L640,240" class="ln"/>
<text x="20" y="250" class="t-sm">Never built during decode:</text>
<text x="20" y="268" class="t-sm" style="fill:var(--ie-waste)">per-head K (T × 128) and V (T × 128), 128 heads</text>
</svg>
<figcaption><b>Where MLA's "decompression" happens: nowhere, during decode.</b> The cache holds
only <span class="k-memory">c_t and k^R_t</span>. The key up-projection moves onto the query
side, and the value up-projection is applied once to the attended latent. Shapes are
DeepSeek-V3's. Prefill, which is compute-bound, can use the explicit form instead, since it
builds K/V for many tokens at once.</figcaption>
</figure>

### What absorption costs

Absorption trades bytes for FLOPs. Each cached token now costs $2 d_c$ FLOPs per head for the
score and $2 d_c$ for the value sum, against $2 d_h$ each in MHA: about 4× more attention
FLOPs per head at $d_c = 4 d_h$. Bytes fall 57× against MHA and 3.6× against GQA-8. The decode
attention kernel moves from intensity 1 (MHA) to about 242 FLOP/byte (step
[3](03-attention-cost.md)), near the H100 ridge, which is why
[FlashMLA](https://github.com/deepseek-ai/FlashMLA) is engineered as a compute-bound decode
kernel. Engines such as vLLM and SGLang use the absorbed form for decode and the explicit
up-projected form for prefill, where compute is what matters.

## Is MLA "just GQA with one group"?

No, for three reasons:

1. **Expressiveness.** In MQA, all heads score against the same key vector. In MLA each head
   scores against its own effective key $W_{UK}^{i} c_t$. The latent is shared, but the
   projection into key space is per head and learned.
2. **What is stored.** GQA stores keys and values, so there is a factor of 2. MLA stores one
   joint latent from which both are derived. Compression is joint.
3. **Where the cost goes.** GQA reduces bytes and FLOPs together. MLA reduces bytes and
   *increases* decode FLOPs, which is a good trade only because decode is bandwidth-bound.

## Comparison

| | MHA | GQA-8 | MQA | MLA (V3) |
|---|---|---|---|---|
| Cached elements / token / layer ($H_q = 128$, $d_h = 128$) | 32,768 | 2,048 | 256 | 576 |
| Relative to MHA | 1 | 1/16 | 1/128 | 1/57 |
| Per-head distinct keys | yes | shared within group | shared by all | yes, via $W_{UK}^{i}$ |
| Decode attention intensity (BF16) | 1 | 16 | 128 | ≈ 242 |
| Retrofit an existing model | — | uptrain about 5% | uptrain | conversion methods exist (MHA2MLA, TransMLA) with fine-tuning |
| Tensor-parallel friendliness | best | good while TP ≤ 8 | KV replicated | latent replicated on every TP rank |

## Experiment

The upstream course's `MultiHeadLatentAttention` module computes MLA in the explicit form: every
step, it up-projects the whole latent cache into per-head K and V. `inference_lab.mla.absorbed_forward`
takes **that same module** and computes the same outputs from its own `k_up_proj`, `v_up_proj` and
`out_proj` weights in absorbed form, optionally with $W_O W_{UV}$ pre-multiplied. The tests check
equality against the module's `forward()` for prefill and decode, with RoPE, to float64 precision
(`test_mla_absorbed_equals_course_module`).

<span class="tag theory">theoretical</span> Counting attention-side FLOPs for one decode step with
the module's shapes ($H = 8$, $d_h = 32$, $d_c = 64$, $d_R = 16$), the explicit form does about
**28× more work** than the absorbed form at both 1K and 16K context. The explicit form's cost
grows with $T \cdot H \cdot d_h \cdot d_c$ (up-projecting the cache), the absorbed form's with
$T \cdot H \cdot d_c$. The notebook [`sota_02_mla_absorption`](../lab/notebooks.md) adds an
optional wall-clock comparison to run on your own machine.

## Failure modes and production notes

- **TP replication.** Because the MLA latent is shared by all heads, splitting heads across 8
  GPUs leaves every GPU holding the *whole* latent cache. Each GPU stores 1/1, not 1/8. For
  this reason SGLang and vLLM serve DeepSeek with data-parallel attention (each GPU owns
  different requests) combined with expert parallelism for the MoE layers.
- **GQA with TP > $H_{kv}$.** Llama-3-70B has 8 KV heads. With TP = 16, KV heads must be
  replicated, and the per-GPU cache stops shrinking.
- **RoPE scaling.** Context-extension methods (YaRN, NTK scaling) act on the decoupled RoPE
  channel only in MLA. See the upstream course [RoPE session](../sessions/07_rope.md).

## Questions

1. Why can $W_{UK}$ be absorbed into the query but RoPE's rotation cannot?
2. Show that MLA's cache is about 1.8× smaller than "MQA with head dim 512" would be, and
   say where the difference comes from.
3. Where in an MLA forward pass is "decompression" conceptually performed during prefill?
   During decode?
4. Why does MLA make tensor parallelism of attention less attractive?

<div class="ie-prov" markdown>

**Provenance**

Source
: MQA: Shazeer, "Fast Transformer Decoding" (2019). GQA: Ainslie et al., EMNLP 2023
  (arXiv:2305.13245). MLA: DeepSeek-AI, DeepSeek-V2 (arXiv:2405.04434) and V3 technical
  report. FlashMLA: github.com/deepseek-ai/FlashMLA (MIT). MHA→MLA conversion: Ji et al.,
  arXiv:2502.14837; TransMLA (Meng et al., 2025).

Derivation
: The absorption steps are the standard rearrangement from the DeepSeek-V2 paper, written out
  here. The intensity and ratio numbers are this site's arithmetic.

Implementation
: `inference_lab.mla.absorbed_forward` (PyTorch, built on the course's `MultiHeadLatentAttention`;
  tested equal to its `forward()`).

Experiment
: FLOP counts only. No timings were run here; the notebook's timing cell is optional.

</div>
