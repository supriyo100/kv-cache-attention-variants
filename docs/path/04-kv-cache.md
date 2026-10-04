---
title: "The KV cache: why it exists, the memory equation, and how fast it fills a GPU"
description: >-
  KV cache explained: why cached keys and values stay valid under causal attention, the KV
  memory equation M = 2·L·H_kv·d_h·s·ΣT with every term defined, worked numbers for Llama and
  DeepSeek models, GPU capacity, growth with unknown output length, and an interactive calculator.
step: 4
layer: model and memory
---

# 4. The KV cache

<p class="lede">Each decode step needs the keys and values of every earlier token. Recomputing
them is quadratic. Storing them is linear, but the store becomes the largest object on the
GPU. This step shows why storing is valid, derives the size of the store, and shows how quickly
it fills an 80 GB card.</p>

## Problem

To produce token $t+1$, every layer needs $K_{1..t}$ and $V_{1..t}$. Where do they come from?

## The naive solution: recompute everything

Run the full sequence of $t$ tokens through the model at every step and keep only the last
position's output. Step $t$ then costs about $2Pt$ FLOPs, and generating $N$ tokens after a
$T_p$-token prompt costs

$$
\sum_{t=T_p}^{T_p+N} 2Pt \;\approx\; 2P\left(T_p N + \tfrac{N^2}{2}\right),
$$

quadratic in output length. Generating 1,000 tokens after a 1,000-token prompt does about 750×
the linear-layer work of a single pass over 2,000 tokens.

## Why caching is valid

Claim: under a causal mask, the key and value of position $j$ in every layer depend only on
tokens $1..j$.

*Proof by induction on layers.* At layer 0, $h^{(0)}_j$ is the embedding of token $j$ (plus
position), so it depends only on token $j$. Suppose every $h^{(\ell)}_i$ with $i \le j$ depends
only on tokens $1..i$. Layer $\ell+1$ computes $h^{(\ell+1)}_j$ from attention over keys and
values at positions $\le j$ (the mask removes the rest), followed by position-wise
operations. So $h^{(\ell+1)}_j$ depends only on $h^{(\ell)}_{1..j}$, and therefore only on
tokens $1..j$. Then $k^{(\ell)}_j = W_K h^{(\ell)}_j$ and $v^{(\ell)}_j = W_V h^{(\ell)}_j$
inherit the property. $\blacksquare$

Appending token $t+1$ therefore leaves every cached $k_j, v_j$ ($j \le t$) unchanged. Each
step computes K and V for the one new position only, appends them, and attends. Per-step
linear-layer cost drops from $2Pt$ to $2P$. Caching trades memory for compute.

The same argument shows what breaks caching: anything that lets an earlier position see a
later one (bidirectional attention, or editing an earlier token) invalidates the cache from that
position onward.

## The memory equation

Each layer stores one key and one value vector per KV head per token:

$$
\boxed{\,M_{\text{KV}} = \underbrace{2}_{K,V} \times L \times H_{kv} \times d_h \times s \times \sum_{i=1}^{B} T_i\,}
$$

| Term | Meaning | Set by |
|---|---|---|
| $2$ | one K and one V | attention type (MLA replaces it, step [5](05-attention-variants.md)) |
| $L$ | layers | model depth |
| $H_{kv}$ | KV heads per layer | MHA: $H_q$; GQA: groups; MQA: 1 |
| $d_h$ | head dimension | model |
| $s$ | bytes per element | KV dtype: BF16 2, FP8 1, INT4 0.5 (+ scales) |
| $\sum T_i$ | tokens currently cached across the batch | workload and scheduler |

Group the model terms into bytes per token, $m_{\text{tok}} = 2 L H_{kv} d_h s$, and the
equation reads $M_{\text{KV}} = m_{\text{tok}} \cdot \sum T_i$. The model fixes
$m_{\text{tok}}$. The workload fixes the sum.

## Worked numbers

<span class="tag theory">theoretical</span> from `kv_math.kv_bytes_per_token` (BF16):

| Model | $L$ | $H_{kv}$ | $d_h$ | $m_{\text{tok}}$ | One 32K sequence |
|---|---|---|---|---|---|
| Llama-2-7B (MHA) | 32 | 32 | 128 | 512 KiB | 16.0 GiB |
| Llama-3-8B (GQA-4) | 32 | 8 | 128 | 128 KiB | 4.0 GiB |
| Llama-3-70B (GQA-8) | 80 | 8 | 128 | 320 KiB | 10.0 GiB |
| DeepSeek-V3 (MLA) | 61 | latent 512 + RoPE 64 | — | 68.6 KiB | 2.14 GiB |

Note the DeepSeek-V3 row. A 671B-parameter model caches *less* per token than an 8B dense
model, because MLA stores a 576-wide latent instead of $2 \times 128 \times 128$ per layer.
(Treating MLA as "one KV head of width $d_c$" with the leading factor 2 overstates it by about
2×, since MLA has no separate V. See step 5.)

## How fast it fills a GPU

Llama-3-8B on one H100 (80 GB): weights take 16 GB. Holding back 10% for activations and
workspace leaves $72 - 16 = 56$ GB, which is **427,246 tokens** of BF16 KV. That is:

| Context per sequence | Concurrent sequences |
|---|---|
| 2,048 | 208 |
| 8,192 | 52 |
| 32,768 | 13 |

FP8 KV doubles every row (854,492 tokens). Llama-3-70B on 4 × H100 leaves about 448K tokens,
also about 13 sequences at 32K. Long-context serving is capacity-limited before it is
compute-limited.

## Try it

<div data-widget="kvcalc"></div>

## Growth, and the unknown final length

<figure class="ie-fig">
<svg viewBox="0 0 760 220" role="img" aria-label="KV memory of one request over time: a jump at prefill, then linear growth during decode; a paged allocator grows in block-sized steps; a contiguous allocator reserves the maximum up front">
<line x1="60" y1="190" x2="730" y2="190" class="ln"/><line x1="60" y1="190" x2="60" y2="15" class="ln"/>
<text x="395" y="212" text-anchor="middle" class="t-sm">time (decode steps)</text>
<text x="20" y="105" transform="rotate(-90 20 105)" text-anchor="middle" class="t-sm">KV memory</text>
<rect x="60" y="30" width="560" height="160" class="waste" style="opacity:.5"/>
<text x="600" y="24" text-anchor="end" class="t-sm" style="fill:var(--ie-waste)">contiguous: reserve prompt + max_tokens up front</text>
<path d="M60,190 L60,130 L620,40" class="ln" style="stroke:var(--ie-ink);stroke-width:1.6"/>
<path d="M60,190 L60,126 L130,126 L130,114 L205,114 L205,102 L280,102 L280,90 L355,90 L355,78 L430,78 L430,66 L505,66 L505,54 L580,54 L580,42 L620,42" class="ln s-memory" style="stroke-width:2"/>
<text x="250" y="140" class="t-sm">actual tokens</text>
<text x="420" y="58" class="t-sm" style="fill:var(--ie-memory)">paged: one block at a time</text>
<line x1="620" y1="40" x2="620" y2="190" class="ln-d"/><text x="624" y="182" class="t-sm">EOS</text>
<text x="66" y="150" class="t-sm">prefill</text>
</svg>
<figcaption><b>One request's KV over its lifetime.</b> Prefill writes $T_p$ tokens at once,
then each decode step adds one. The server learns the final length only at EOS. A contiguous
allocator must reserve the worst case (shaded <span class="k-waste">waste</span>). A paged
allocator tracks the line within one block (<span class="k-memory">staircase</span>). Steps 6
and 7 quantify the difference.</figcaption>
</figure>

## Implementation

[`inference_lab/kv_math.py`](https://github.com/supriyo100/kv-cache-attention-variants/blob/main/src/inference_lab/kv_math.py)
holds the equation, MLA's variant, the model shapes and `max_concurrent_tokens`. Tests pin
Llama-3-8B at exactly 131,072 bytes per token, DeepSeek-V3 at $61 \times 576 \times 2$, and
the MHA : GQA : MQA ratio at 32 : 8 : 1.

## Trade-offs and failure modes

- **Memory for compute.** Caching is almost always right. The exception is very short
  generations on memory-starved hardware, where recomputing a short prefix can beat storing it.
- **Precision.** FP8 KV halves memory and bandwidth. Accuracy impact is usually small but
  model-dependent, and INT4/INT2 needs care (step [11](11-long-context.md)).
- **Static allocation (pre-2023 engines).** Reserving `max_seq_len` per request wastes most of
  the cache. Step 6 measures it.
- **Batching with ragged lengths.** Padding every sequence to the longest in the batch wastes
  both memory and attention FLOPs.

## Production notes

vLLM profiles a forward pass at startup, multiplies HBM by `gpu_memory_utilization` (default
0.9), subtracts weights and peak activations, and divides the rest into KV blocks. The log line
reporting the number of GPU blocks is the KV capacity in the equation above. SGLang does the
same with `mem_fraction_static`. Both support FP8 KV (`kv_cache_dtype=fp8`).

## Questions

1. Prove that caching would be invalid for an encoder with bidirectional attention.
2. A request has a 6,000-token prompt and `max_tokens=4096` but stops after 300 tokens. How much
   memory does a static allocator waste, as a fraction of what it reserved?
3. With Llama-3-8B on one H100, you must serve 64 concurrent 16K-context chats. What are your
   options? Use the calculator.

<div class="ie-prov" markdown>

**Provenance**

Source
: KV caching is standard autoregressive-decoding practice; its memory pressure is analysed in
  Pope et al., "Efficiently Scaling Transformer Inference" (2022) and Kwon et al. (vLLM, 2023).
  Model shapes come from each model's `config.json`.

Derivation
: The induction argument, the recompute cost and the capacity tables are this site's.

Implementation
: `inference_lab.kv_math` and the in-browser calculator (the same formulas).

Experiment
: None. Capacity numbers are arithmetic with a stated 10% reserve.

</div>
