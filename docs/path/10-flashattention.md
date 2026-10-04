---
title: "FlashAttention explained: IO-aware attention, tiling and the online softmax"
description: >-
  Why naive attention is limited by HBM traffic and T×T memory, how FlashAttention tiles Q, K
  and V through SRAM with an exact online softmax, the IO complexity, what FlashAttention-2 and -3
  changed, split-KV decoding (FlashDecoding), FlashInfer, and why FlashAttention does not solve
  KV capacity.
step: 10
layer: kernel
---

# 10. FlashAttention

<p class="lede">Steps 4–9 decided what to store and where. This step is about computing attention
over it: a kernel problem. FlashAttention computes exact attention without ever writing the
T×T score matrix to GPU memory. It is often summarized as "faster attention". The more
precise statement is that it turns attention's memory from quadratic to linear and removes
round trips to HBM, using one algebraic trick, the online softmax.</p>

## Problem

A GPU has two memories that matter here. **HBM**: 80 GB at about 3.35 TB/s on an H100.
**On-chip SRAM** (shared memory and registers): about 228 KB per SM, 132 SMs, an order of
magnitude more bandwidth. Naive attention lives in HBM:

```
S = Q @ K.T / sqrt(d)   # write T×T to HBM
P = softmax(S)          # read T×T, write T×T
O = P @ V               # read T×T
```

Each stage is a separate kernel that reads its input from HBM and writes its output back.

## The IO count

Counting elements moved, for one head with $T$ tokens and head dim $d$:

$$
Q_{\text{naive}} = \underbrace{2Td}_{\text{read }Q,K} + \underbrace{T^2 + T^2}_{\text{write, read }S} + \underbrace{T^2 + T^2}_{\text{write, read }P} + \underbrace{2Td}_{\text{read }V\text{, write }O} = 4Td + 4T^2.
$$

At $T = 16{,}384$, $d = 128$ in FP16 that is 2,064 MiB per head per layer, and the $4T^2$
term is over 99% of it. Worse, $S$ and $P$ must exist in memory: 512 MiB each for one head.

## Naive fix, and why it fails

"Fuse the three stages into one kernel." The obstacle is the softmax. Row $i$ of $P$ needs
$\max_j S_{ij}$ and $\sum_j e^{S_{ij}}$ over **all** $j$ before any $P_{ij}$ is known. A
fused kernel would have to hold an entire row of $T$ scores on chip, which does not fit at
long context.

## Core idea: the online softmax

Process keys in blocks and keep three running statistics per query row: the max $m$, the
denominator $\ell$, and an unnormalized output $\tilde o$. After block $j$ with scores
$S_j$ and values $V_j$:

$$
\begin{aligned}
m^{(j)} &= \max\!\big(m^{(j-1)},\ \operatorname{rowmax} S_j\big),\\
\ell^{(j)} &= e^{\,m^{(j-1)} - m^{(j)}}\,\ell^{(j-1)} + \operatorname{rowsum} e^{\,S_j - m^{(j)}},\\
\tilde o^{(j)} &= e^{\,m^{(j-1)} - m^{(j)}}\,\tilde o^{(j-1)} + e^{\,S_j - m^{(j)}}\,V_j,
\end{aligned}
\qquad\text{and finally}\quad O = \tilde o^{(n)} / \ell^{(n)}.
$$

**Why it is exact.** By induction, after block $j$:
$\ell^{(j)} = \sum_{k \le j} \sum e^{S_k - m^{(j)}}$ and
$\tilde o^{(j)} = \sum_{k\le j} e^{S_k - m^{(j)}} V_k$. When the max rises from $m$ to $m'$,
every earlier term should have been scaled by $e^{-m'}$ instead of $e^{-m}$. Multiplying the
accumulated sums by $e^{m - m'}$ applies exactly that correction. After the last block,
$\tilde o / \ell = \sum_k e^{S_k - m} V_k \big/ \sum_k e^{S_k - m} = \operatorname{softmax}(S)V$.
No approximation is involved. The kernel also keeps $\text{LSE} = m + \log \ell$ for the
backward pass and for merging (below).

## Tiling

<figure class="ie-fig">
<svg viewBox="0 0 760 260" role="img" aria-label="FlashAttention tiling: a block of Q rows is loaded into SRAM once; blocks of K and V stream through SRAM; the running max, sum and output accumulate on chip; only O is written to HBM">
<rect x="20" y="20" width="250" height="220" class="box" style="fill:var(--ie-paper-2)"/><text x="145" y="40" text-anchor="middle" class="t-b">HBM (slow, large)</text>
<rect x="40" y="60" width="40" height="160" class="compute"/><text x="60" y="235" text-anchor="middle" class="t-mono">Q</text>
<rect x="40" y="100" width="40" height="30" class="compute" style="stroke-width:2.5"/>
<rect x="100" y="60" width="40" height="160" class="memory"/><text x="120" y="235" text-anchor="middle" class="t-mono">K</text>
<rect x="160" y="60" width="40" height="160" class="memory"/><text x="180" y="235" text-anchor="middle" class="t-mono">V</text>
<rect x="100" y="60" width="40" height="30" class="memory" style="stroke-width:2.5"/><rect x="160" y="60" width="40" height="30" class="memory" style="stroke-width:2.5"/>
<rect x="215" y="60" width="40" height="160" class="box"/><text x="235" y="235" text-anchor="middle" class="t-mono">O</text>
<rect x="400" y="20" width="340" height="220" class="box" style="fill:var(--ie-paper-2)"/><text x="570" y="40" text-anchor="middle" class="t-b">SRAM, one thread block (fast, ~228 KB/SM)</text>
<rect x="420" y="60" width="80" height="30" class="compute"/><text x="460" y="80" text-anchor="middle" class="t-sm">Q_i tile</text>
<rect x="520" y="60" width="80" height="30" class="memory"/><text x="560" y="80" text-anchor="middle" class="t-sm">K_j, V_j tile</text>
<rect x="420" y="110" width="180" height="40" class="waste" style="opacity:.75"/><text x="510" y="135" text-anchor="middle" class="t-sm">S_ij tile (B_r × B_c), never leaves</text>
<rect x="620" y="60" width="100" height="90" class="compute" style="opacity:.6"/><text x="670" y="90" text-anchor="middle" class="t-sm">m, ℓ, õ</text><text x="670" y="108" text-anchor="middle" class="t-sm">running stats</text><text x="670" y="124" text-anchor="middle" class="t-sm">(rescaled)</text>
<text x="420" y="180" class="t-sm">for each K_j, V_j block: load, score, update m, ℓ, õ</text>
<text x="420" y="200" class="t-sm">after the last block: write O_i = õ / ℓ and LSE_i</text>
<path d="M80,115 C200,90 300,70 420,75" class="ln"/><text x="300" y="66" class="t-sm">load once</text>
<path d="M140,75 C260,110 380,110 520,80" class="ln s-memory" style="stroke:var(--ie-memory)"/><text x="300" y="118" class="t-sm" style="fill:var(--ie-memory)">stream every block</text>
<path d="M670,150 C600,230 350,240 255,150" class="ln-d"/><text x="430" y="232" class="t-sm">write O once</text>
</svg>
<figcaption><b>What stays on chip.</b> Each thread block owns a tile of query rows. K/V tiles
stream past it, and the <span class="k-waste">score tile</span> lives only in SRAM. HBM
traffic is Q and O once, plus K and V once per query tile. No T×T tensor is ever written.</figcaption>
</figure>

With SRAM of $M$ elements and query tiles of $B_r \approx M/(4d)$ rows, K and V are streamed
$\lceil T/B_r \rceil$ times:

$$
Q_{\text{flash}} \approx 2Td + \frac{T}{B_r}\cdot 2Td = \Theta\!\left(\frac{T^2 d^2}{M}\right),
\qquad \frac{Q_{\text{naive}}}{Q_{\text{flash}}} \approx \frac{M}{2d^2}\ \text{for large }T.
$$

| $T$ (FP16, $d=128$, $M = 10^5$ elements) | Naive HBM traffic | Tiled HBM traffic |
|---|---|---|
| 1,024 | 9.0 MiB | 3.5 MiB |
| 4,096 | 132 MiB | 46 MiB |
| 16,384 | 2,064 MiB | 688 MiB |

<span class="tag theory">theoretical</span>, from `flash.io_naive` and `flash.io_flash`. The
traffic saving at $d = 128$ is about 3× by this count, and larger for $d = 64$ (about 12×).
The headline wins are elsewhere: **memory drops from $O(T^2)$ to $O(T)$**, which is what made
64K–1M contexts possible, and the three kernels become one, with no launch gaps and no
intermediate writes. The paper reports 2–4× end-to-end speedups over standard implementations
at the time.

## FlashAttention-2, -3, FlashDecoding and FlashInfer

| | What it changed | Why it mattered |
|---|---|---|
| FlashAttention (2022) | tiling + online softmax; recompute in backward | exact attention in $O(T)$ memory |
| FlashAttention-2 (2023) | parallelize over sequence length too; split work across warps by Q, not K; fewer non-matmul FLOPs | about 2× over v1; roughly 50–73% of A100 peak in forward |
| FlashAttention-3 (2024) | Hopper features: asynchronous TMA loads, warp specialization, overlapping softmax with matmul, FP8 with incoherent processing | about 75% of H100 BF16 peak; FP8 support |
| FlashDecoding / split-KV (2023) | at decode, $Q$ is one row per sequence, so split the **KV** across thread blocks and merge partial results | fills all SMs when batch × heads is small and context is long |
| FlashInfer (2024–) | library of paged/ragged attention kernels, JIT-specialized variants, cascade attention for shared prefixes | default attention backend in SGLang and an option in vLLM |

### Merging partial results exactly

Split-KV decoding computes attention of one query over disjoint chunks $A$ and $B$ of the
context, giving $(o_A, \text{LSE}_A)$ and $(o_B, \text{LSE}_B)$. They combine exactly:

$$
\text{LSE} = \log\!\big(e^{\text{LSE}_A} + e^{\text{LSE}_B}\big),\qquad
o = e^{\text{LSE}_A - \text{LSE}}\, o_A + e^{\text{LSE}_B - \text{LSE}}\, o_B .
$$

This one identity powers FlashDecoding, context parallelism (step [12](12-scaling-out.md)),
and FlashInfer's cascade attention. In cascade attention, many requests attend to the same
shared prefix (step 8) in one batched pass, then each attends to its own suffix, and the two
results are merged.

## FlashAttention vs PagedAttention

They are complementary and solve different problems at different layers:

| | FlashAttention | PagedAttention |
|---|---|---|
| Layer | kernel | cache management |
| Problem | HBM traffic and $O(T^2)$ memory of the score matrix | fragmentation of KV storage |
| Changes | how attention is computed | where K/V blocks live |
| Reduces KV cache size? | **no** | no (it reduces waste around it) |
| Used together? | yes: paged FlashAttention/FlashInfer kernels read K/V through block tables | |

**Why FlashAttention does not solve KV capacity.** It never stores the T×T matrix, but K and V
must still be stored for every token, in every layer, for every sequence. The cache equation in
step 4 is unchanged. Only architecture (step 5), precision, eviction (step 11) and sharing
(step 8) change it.

## Implementation

[`inference_lab/flash.py`](https://github.com/supriyo100/kv-cache-attention-variants/blob/main/src/inference_lab/flash.py)
is written in PyTorch on `(batch, heads, seq, head_dim)` tensors, the course's layout. It has
`tiled_attention` (the online-softmax recurrence above, with causal tile skipping and FP32
accumulation for FP16/BF16 inputs), `merge_states` (the LSE merge), `split_kv_attention`
(FlashDecoding semantics) and the IO counters. Tests check tiled and split-KV results against
`torch.nn.functional.scaled_dot_product_attention`, with and without a causal mask.

## Failure modes

- **Small head dims, big tiles.** Tile shapes are tuned per architecture and head dim. An
  untuned shape can be slower than a vendor GEMM-based path for short sequences.
- **Numerics.** The rescaling must be done in FP32. FP8 attention needs per-block scaling
  (FA-3's incoherent processing) to keep error acceptable.
- **Decode is still bandwidth-bound.** FlashAttention does not reduce the bytes of K/V a
  decode step must read. Only GQA/MLA/quantization do.

## Questions

1. Show that the online-softmax update is associative, so blocks can be processed in any
   order and merged.
2. Why does FlashDecoding help at batch 1 with 128K context but not at batch 256 with 2K?
3. A colleague says "we use FlashAttention, so we don't need PagedAttention." Respond.

<div class="ie-prov" markdown>

**Provenance**

Source
: Dao et al., FlashAttention (NeurIPS 2022, arXiv:2205.14135); Dao, FlashAttention-2
  (arXiv:2307.08691); Shah et al., FlashAttention-3 (arXiv:2407.08608); github.com/Dao-AILab/flash-attention
  (BSD-3). Online softmax: Milakov and Gimelshein (2018). FlashDecoding (Dao et al., 2023
  blog). FlashInfer: Ye et al. (arXiv:2501.01005), github.com/flashinfer-ai/flashinfer.

Derivation
: Exactness proof and IO table written for this site.

Implementation
: `inference_lab.flash` (PyTorch reference loops, tested against SDPA).

Experiment
: No kernels were timed. The `13_flash_attention_io_model` and
  `sota_03_flash_decoding_lse_merge` notebooks reproduce the checks when you run them.

</div>
