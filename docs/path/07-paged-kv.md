---
title: "PagedAttention explained: virtual memory for the LLM KV cache"
description: >-
  How paged KV caches work: logical vs physical blocks, block tables, the virtual-memory analogy,
  how attention kernels gather scattered blocks, on-demand allocation, and why paging removes
  external fragmentation but not internal fragmentation, with an interactive allocator.
step: 7
layer: cache management
---

# 7. Paged KV

<p class="lede">Operating systems solved this problem fifty years ago. Programs see contiguous
virtual addresses, and a page table maps them onto whatever physical frames are free. A paged KV
cache does the same for tokens. This step shows the mapping, what it costs inside the attention
kernel, and why "32K context" no longer means "32K slots reserved".</p>

## Problem

From step 6: contiguous KV allocation wastes a third or more of the pool to reservations and
holes, and rejects requests while memory sits free. We want three things: allocate only what
is used, use any free memory regardless of where it is, and keep attention fast.

## Core idea

Split each sequence's KV into **logical blocks** of $B$ tokens. Keep a **block table** per
sequence that maps logical block $i$ to a **physical block** id in one shared pool. Blocks are
allocated one at a time, when a sequence fills its last block.

| Operating system | Paged KV cache |
|---|---|
| process | sequence (request) |
| virtual page | logical block of $B$ tokens |
| physical frame | physical KV block in HBM |
| page table | block table |
| page fault, then allocate frame | last block full, then allocate block |
| shared pages (fork, libraries) | shared prefix blocks (step [8](08-prefix-sharing.md)) |
| copy-on-write | copy-on-write of a shared partial block |
| swap to disk | swap KV to CPU memory, or recompute (step [9](09-scheduling.md)) |

## The math: address translation

For token position $p$ of sequence $r$:

$$
\text{logical block} = \lfloor p / B \rfloor,\qquad
\text{offset} = p \bmod B,\qquad
\text{slot}(r, p) = \text{table}_r\big[\lfloor p/B\rfloor\big]\cdot B + (p \bmod B).
$$

A sequence of $T$ tokens holds $\lceil T/B \rceil$ blocks, and the pool holds
$\lfloor M_{\text{KV}} / (B\, m_{\text{tok}}) \rfloor$ blocks in total. The block table costs
4 bytes per block: for a 32K sequence at $B=16$, 2,048 entries or 8 KiB, against 4 GiB of KV
for Llama-3-8B. The metadata is negligible.

<figure class="ie-fig">
<svg viewBox="0 0 760 290" role="img" aria-label="Logical blocks 0 to 3 of a sequence, contiguous, map through a block table to physical blocks 7, 19, 4 and 22 in a pool shared with other sequences">
<text x="20" y="24" class="t-b">Logical view: one sequence, contiguous</text>
<rect x="20" y="36" width="150" height="36" class="memory"/><text x="95" y="59" text-anchor="middle" class="t-sm">tokens 0–15</text>
<rect x="176" y="36" width="150" height="36" class="memory"/><text x="251" y="59" text-anchor="middle" class="t-sm">tokens 16–31</text>
<rect x="332" y="36" width="150" height="36" class="memory"/><text x="407" y="59" text-anchor="middle" class="t-sm">tokens 32–47</text>
<rect x="488" y="36" width="150" height="36" class="waste" style="opacity:.6"/><rect x="488" y="36" width="56" height="36" class="memory"/><text x="563" y="59" text-anchor="middle" class="t-sm">48–53 (+10 free)</text>
<text x="20" y="104" class="t-b">Block table</text>
<g class="t-mono">
<rect x="120" y="88" width="80" height="26" class="box"/><text x="160" y="106" text-anchor="middle" class="t-mono">0 → 7</text>
<rect x="210" y="88" width="80" height="26" class="box"/><text x="250" y="106" text-anchor="middle" class="t-mono">1 → 19</text>
<rect x="300" y="88" width="80" height="26" class="box"/><text x="340" y="106" text-anchor="middle" class="t-mono">2 → 4</text>
<rect x="390" y="88" width="80" height="26" class="box"/><text x="430" y="106" text-anchor="middle" class="t-mono">3 → 22</text>
</g>
<text x="20" y="150" class="t-b">Physical pool (HBM), shared by all sequences</text>
<g>
<rect x="20" y="162" width="26" height="26" class="free"/><rect x="50" y="162" width="26" height="26" class="box" style="opacity:.5"/><rect x="80" y="162" width="26" height="26" class="box" style="opacity:.5"/><rect x="110" y="162" width="26" height="26" class="free"/><rect x="140" y="162" width="26" height="26" class="memory"/><rect x="170" y="162" width="26" height="26" class="box" style="opacity:.5"/><rect x="200" y="162" width="26" height="26" class="free"/><rect x="230" y="162" width="26" height="26" class="memory"/><rect x="260" y="162" width="26" height="26" class="box" style="opacity:.5"/><rect x="290" y="162" width="26" height="26" class="free"/><rect x="320" y="162" width="26" height="26" class="box" style="opacity:.5"/><rect x="350" y="162" width="26" height="26" class="free"/>
<rect x="20" y="198" width="26" height="26" class="box" style="opacity:.5"/><rect x="50" y="198" width="26" height="26" class="free"/><rect x="80" y="198" width="26" height="26" class="box" style="opacity:.5"/><rect x="110" y="198" width="26" height="26" class="free"/><rect x="140" y="198" width="26" height="26" class="box" style="opacity:.5"/><rect x="170" y="198" width="26" height="26" class="free"/><rect x="200" y="198" width="26" height="26" class="box" style="opacity:.5"/><rect x="230" y="198" width="26" height="26" class="memory"/><rect x="260" y="198" width="26" height="26" class="free"/><rect x="290" y="198" width="26" height="26" class="box" style="opacity:.5"/><rect x="320" y="198" width="26" height="26" class="memory"/><rect x="350" y="198" width="26" height="26" class="free"/>
</g>
<text x="153" y="180" text-anchor="middle" class="t-sm">4</text><text x="243" y="180" text-anchor="middle" class="t-sm">7</text><text x="243" y="216" text-anchor="middle" class="t-sm">19</text><text x="333" y="216" text-anchor="middle" class="t-sm">22</text>
<path d="M160,114 L243,162" class="ln s-memory" style="stroke:var(--ie-memory)"/><path d="M250,114 L243,198" class="ln s-memory" style="stroke:var(--ie-memory)"/>
<path d="M340,114 L153,162" class="ln s-memory" style="stroke:var(--ie-memory)"/><path d="M430,114 L333,198" class="ln s-memory" style="stroke:var(--ie-memory)"/>
<text x="400" y="180" class="t-sm"><tspan style="fill:var(--ie-memory)">■</tspan> this sequence   ■ other sequences (grey)   □ free</text>
<text x="400" y="200" class="t-sm">Order lives in the table, not in memory.</text>
<text x="400" y="220" class="t-sm">Any free block can serve any sequence.</text>
<text x="20" y="262" class="t-sm">slot(token 37) = table[37 // 16] · 16 + 37 % 16 = 4 · 16 + 5 = 69</text>
</svg>
<figcaption><b>Logical contiguity, physical scatter.</b> The model and the scheduler see tokens
0–53 in order. HBM holds them in blocks 7, 19, 4 and 22, interleaved with other requests' blocks.
The only waste is the 10 empty slots of the last block.</figcaption>
</figure>

## Why this removes external fragmentation

External fragmentation requires *variable-size* contiguous requests. In a paged pool every
allocation is exactly one block, and every free block is equally good. If $k$ blocks are free,
any request needing $\le k$ blocks fits. By construction there are no unusable holes at block
granularity. Internal fragmentation remains (step 6): at most $B-1$ slots per sequence.

## Why a 32K context no longer needs a 32K allocation

Admission needs only $\lceil T_p / B \rceil$ blocks for the prompt. Each later block is
allocated at the moment the previous one fills, one block per $B$ decode steps. A request
capped at 32K that stops at 900 tokens never holds more than 57 blocks. The scheduler's job
changes from "reserve the worst case" to "make sure growth can be served, and have a plan
when it can't" (step 9).

## What it costs in the kernel

The attention kernel can no longer read K/V with one base pointer and a stride. For each block
of keys it loads the physical block id from the table, then reads that block:

```python
# decode attention for one query over a paged cache (reference semantics)
for logical_block, phys in enumerate(block_table[seq]):
    k_tile = K_pool[phys]            # (B, H_kv, d_h): contiguous inside the block
    v_tile = V_pool[phys]
    scores = q @ k_tile.T            # masked for the last, partial block
    ...                              # online-softmax accumulate (step 10)
```

Inside a block, memory is contiguous, so each gather is a fully coalesced read of
$B \cdot H_{kv} \cdot d_h \cdot s$ bytes: 32 KiB of K (and as much of V) per layer for
Llama-3-8B at $B = 16$. The
extra cost is one table lookup per block and less freedom in tile shapes. The vLLM paper reports
the paged kernel within roughly 20–26% of a contiguous kernel's attention latency, a cost
repaid many times over by larger batches. Later kernels (FlashInfer, FlashAttention-2/3 paged
variants) narrowed the gap further.

## Try it

<div data-widget="allocator"></div>

## Implementation

[`inference_lab/paged.py`](https://github.com/supriyo100/kv-cache-attention-variants/blob/main/src/inference_lab/paged.py)
is about 150 lines: a `BlockAllocator` (free list and refcounts) and a `PagedKVCache` with
`add` (prefill), `append_token` (decode, allocating on block boundaries), `fork` (share all
blocks), `free`, and `physical_slot` (address translation). `check_invariants` asserts that
refcounts equal table references and that the free list holds exactly the zero-refcount
blocks. A randomized test drives 400 mixed operations and checks the invariants after each one.

## Trade-offs and failure modes

- **Indirection in every kernel.** All attention kernels (prefill, decode, speculative
  verification) must understand block tables. Custom kernels that assume contiguous K/V can't
  be used directly.
- **Running out mid-flight.** Growth is optimistic, so a decode step can need a block when
  none is free. The scheduler must preempt (step 9).
- **CPU overhead.** Block tables are built and copied to the GPU every iteration. At large
  batch sizes this became a measurable fraction of step time, one motivation for vLLM V1's
  rewritten scheduler and persistent batch state.
- **Alternatives exist.** vAttention (Prabhu et al., 2024) keeps a contiguous *virtual* KV
  tensor and maps physical pages underneath with CUDA virtual-memory APIs, so unmodified
  kernels work.

## Questions

1. Why is external fragmentation zero at block granularity, and what is the analogue of
   external fragmentation one level up (for example, at the level of whole GPUs)?
2. A sequence of length 1,000 with $B=16$: how many blocks, which block holds token 999, and
   at what offset?
3. Why does paging make beam search and parallel sampling cheaper? (Preview of step 8.)

<div class="ie-prov" markdown>

**Provenance**

Source
: Kwon et al., SOSP 2023 (arXiv:2309.06180), vLLM (github.com/vllm-project/vllm,
  Apache-2.0). vAttention: Prabhu et al., arXiv:2405.04437. FlashInfer: Ye et al., arXiv:2501.01005.

Derivation
: The address translation and table-size arithmetic are this site's.

Implementation
: `inference_lab.paged` (tested: random workload invariants, table translation,
  copy-on-write).

Experiment
: The in-browser allocator mirrors `paged.py`. No kernel timings were run here; the 20–26%
  figure is the vLLM paper's.

</div>
