---
title: "KV cache fragmentation: reservation, internal and external waste"
description: >-
  Why contiguous KV cache allocation wastes GPU memory: reservation waste from unknown output
  length, external fragmentation from variable-size holes, and internal fragmentation inside
  blocks (waste < block size), with a simulation comparing contiguous and paged allocation.
step: 6
layer: cache management
---

# 6. Fragmentation

<p class="lede">The cache equation says how many bytes a workload needs. It doesn't say how
many bytes you must hold to provide them. Between the two sits fragmentation: memory that is
allocated or free but cannot hold the tokens you need. This step separates three kinds of
waste. Paging removes two of them and bounds the third.</p>

## Problem

A server with 64K token slots of KV runs requests whose final lengths it cannot know in
advance. It allocates each request's KV as one contiguous region, as a simple tensor-per-request
engine would. How much of that memory ever holds a real token?

## The naive solution: one contiguous region per request

Give each request a contiguous tensor of `prompt + max_tokens` slots. Attention kernels like
this: one base pointer and a stride. It fails in three ways.

### 1. Reservation waste (over-allocation)

The request may stop long before `max_tokens`. At decode step $n$, a request with prompt
$T_p$ and cap $N_{\max}$ uses $T_p + n$ of its $T_p + N_{\max}$ slots. Over a lifetime that
ends at $N$ tokens, the mean fraction in use is

$$
u = \frac{T_p + N/2}{T_p + N_{\max}}.
$$

A 500-token prompt with `max_tokens=2048` that stops after 200 tokens uses on average
$600 / 2548 = 24\%$ of its reservation. The other 76% is reserved but empty for the request's
whole life.

### 2. External fragmentation

Requests reserve different sizes and finish at different times, so free memory breaks into
holes of different sizes:

```
[ A: 2,548 ][ free 900 ][ B: 1,300 ][ free 1,700 ][ C: 3,100 ][ free 600 ]
```

There are 3,200 free slots, but a request needing 2,000 contiguous slots is rejected.
Compaction would fix it, but moving gigabytes of live KV between steps stalls every request.

### 3. Internal fragmentation

Once memory comes in fixed-size blocks of $B$ tokens (the paged fix in step
[7](07-paged-kv.md)), a sequence of $T$ tokens occupies $\lceil T/B \rceil$ blocks and leaves

$$
w(T) = \lceil T/B\rceil B - T = (-T) \bmod B, \qquad 0 \le w < B
$$

empty slots in its last block. With $B = 16$ and $T = 18$: block 1 holds 16 tokens, block 2
holds 2 tokens plus 14 empty slots. If lengths are spread evenly modulo $B$, the expected waste
is $(B-1)/2$ slots per sequence. That is 7.5 slots for $B = 16$, about 1 MiB per sequence
for Llama-3-8B, and it is bounded by $B - 1$ no matter how long the sequence gets.

<figure class="ie-fig">
<svg viewBox="0 0 760 250" role="img" aria-label="Top: contiguous allocation with large reservations, mostly empty, and holes between them. Bottom: paged allocation where each request's blocks are full except the last">
<text x="20" y="22" class="t-b">Contiguous, reserve prompt + max_tokens</text>
<rect x="20" y="32" width="160" height="34" class="waste" style="opacity:.6"/><rect x="20" y="32" width="52" height="34" class="memory"/><text x="26" y="54" class="t-sm">A</text>
<rect x="180" y="32" width="60" height="34" class="free"/><text x="210" y="54" text-anchor="middle" class="t-sm">hole</text>
<rect x="240" y="32" width="90" height="34" class="waste" style="opacity:.6"/><rect x="240" y="32" width="64" height="34" class="memory"/><text x="246" y="54" class="t-sm">B</text>
<rect x="330" y="32" width="110" height="34" class="free"/><text x="385" y="54" text-anchor="middle" class="t-sm">hole</text>
<rect x="440" y="32" width="200" height="34" class="waste" style="opacity:.6"/><rect x="440" y="32" width="40" height="34" class="memory"/><text x="446" y="54" class="t-sm">C</text>
<rect x="640" y="32" width="100" height="34" class="free"/><text x="690" y="54" text-anchor="middle" class="t-sm">hole</text>
<text x="20" y="86" class="t-sm"><tspan style="fill:var(--ie-memory)">■</tspan> tokens  <tspan style="fill:var(--ie-waste)">■</tspan> reserved, never or not yet used  □ free but too small for the next request (external)</text>
<text x="20" y="130" class="t-b">Paged, blocks of B tokens, any free block fits</text>
<g>
<rect x="20" y="140" width="40" height="34" class="memory"/><rect x="64" y="140" width="40" height="34" class="memory"/><rect x="108" y="140" width="40" height="34" class="waste" style="opacity:.6"/><rect x="108" y="140" width="12" height="34" class="memory"/>
<rect x="152" y="140" width="40" height="34" class="memory"/><rect x="196" y="140" width="40" height="34" class="memory"/><rect x="240" y="140" width="40" height="34" class="memory"/><rect x="284" y="140" width="40" height="34" class="waste" style="opacity:.6"/><rect x="284" y="140" width="30" height="34" class="memory"/>
<rect x="328" y="140" width="40" height="34" class="memory"/><rect x="372" y="140" width="40" height="34" class="waste" style="opacity:.6"/><rect x="372" y="140" width="22" height="34" class="memory"/>
<rect x="416" y="140" width="40" height="34" class="free"/><rect x="460" y="140" width="40" height="34" class="free"/><rect x="504" y="140" width="40" height="34" class="free"/><rect x="548" y="140" width="40" height="34" class="free"/><rect x="592" y="140" width="40" height="34" class="free"/><rect x="636" y="140" width="40" height="34" class="free"/><rect x="680" y="140" width="40" height="34" class="free"/>
</g>
<text x="84" y="192" text-anchor="middle" class="t-sm">A: 3 blocks</text><text x="238" y="192" text-anchor="middle" class="t-sm">B: 4 blocks</text><text x="370" y="192" text-anchor="middle" class="t-sm">C: 2 blocks</text><text x="568" y="192" text-anchor="middle" class="t-sm">free: any request can use any of these</text>
<text x="20" y="226" class="t-sm">Waste is only in each request's last block (<tspan style="fill:var(--ie-waste)">■</tspan>): fewer than B slots per request. No holes, no reservations.</text>
</svg>
<figcaption><b>Same three requests, two allocators.</b> Contiguous allocation wastes the
reservation <i>and</i> the holes between reservations. Paged allocation wastes at most one
partially filled block per request. Blocks need not be adjacent; the block table in step 7
keeps the order.</figcaption>
</figure>

## Fragmentation is not duplication

Two different problems are often mixed up:

| | Fragmentation | Duplication |
|---|---|---|
| What is wrong | slots hold no token | slots hold a token another request already stored |
| Example | 14 empty slots in a last block | 50 users each storing the same 2,000-token system prompt |
| Fixed by | paging (steps 6–7) | prefix sharing (step [8](08-prefix-sharing.md)) |
| Does paging alone fix it? | yes (external), bounded (internal) | no |

## Experiment

<span class="tag measured">measured</span> (a deterministic simulation on CPU, not GPU memory):
`inference_lab.fragmentation.simulate` replays one trace of 300 requests through both
allocators over a 65,536-slot pool. Each request has a prompt of 64–1,024 tokens, a declared
`max_tokens` cap drawn from {256, 512, 1024, 2048}, and a true output length below its cap.
One decode token per step for every running request. Rejected requests are dropped, so both
allocators see the same arrivals.

| | Contiguous (reserve prompt + cap) | Paged ($B = 16$) |
|---|---|---|
| Admitted | 127 | 274 |
| Rejected, enough free memory but no hole large enough (external) | 169 | 0 |
| Rejected, not enough memory | 4 | 26 |
| Mean slot utilization (live tokens / allocated) | 65.5% | 99.4% |
| Mean waste, slots | 15,525 (reservation) | 268 (internal) |
| Peak live tokens | 37,564 | 65,104 |

Contiguous allocation turned away 169 requests while enough total memory was free: external
fragmentation, in the numbers. Paging admitted more than twice as many requests and pushed the
pool to 99% utilization. Its remaining waste, 268 slots, is about 7.5 slots per running
sequence, the $(B-1)/2$ the formula predicts.

Simplifications, stated: paged mode stops a request early if the pool is empty instead of
preempting (step 9 models preemption), and contiguous mode does not compact.

## Choosing the block size

| Smaller $B$ | Larger $B$ |
|---|---|
| less internal waste, $(B-1)/2$ per sequence | fewer blocks, smaller block tables |
| finer-grained prefix sharing (step 8) | more contiguous memory per gather, so better kernel efficiency |
| more metadata and indirection | coarser sharing: only full blocks are shared |

vLLM's default is 16 tokens. FlashInfer's paged kernels support page size 1. SGLang's radix
cache has historically managed memory at token granularity. The waste is already small at 16,
so the choice is driven by kernel efficiency and sharing granularity more than by memory.

## Failure modes

- **Paging does not fix reservation by itself.** An engine that pre-allocates all
  `max_tokens` blocks at admission still wastes them. Paging pays off only with on-demand
  growth.
- **Growth can fail mid-request.** On-demand growth means a running request can find the
  pool empty. That is the price of not reserving, and step 9 handles it.
- **Many tiny sequences.** With thousands of short requests, $(B-1)/2$ per sequence adds up,
  and smaller blocks or token-level pools win.

## Questions

1. Why does a buddy allocator, or another general-purpose allocator, not solve this?
2. Derive the expected internal waste if output lengths are always multiples of 8 and
   $B = 16$.
3. Prefix sharing reduces memory, but does it reduce fragmentation? Explain with the table
   above.

<div class="ie-prov" markdown>

**Provenance**

Source
: Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention" (SOSP 2023)
  first quantified reservation, internal and external fragmentation in LLM serving. Classical
  OS fragmentation theory (for example Silberschatz, *Operating System Concepts*).

Derivation
: The utilization formula, the $(B-1)/2$ expectation and the duplication-vs-fragmentation
  framing are this site's.

Implementation
: `inference_lab.fragmentation` (`ContiguousPool` first-fit with coalescing, `simulate`).
  Tested in `test_paged_beats_contiguous_on_same_trace` and `test_internal_waste_below_block_size`.

Experiment
: Deterministic simulation, seed 1, run on CPU. Reproduce with
  `notebooks/07_external_internal_fragmentation.ipynb`.

</div>
