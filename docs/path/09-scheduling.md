---
title: "LLM scheduling: continuous batching, chunked prefill, admission and preemption"
description: >-
  How LLM servers decide which requests run each iteration: static vs continuous (iteration-level)
  batching, memory-aware admission, chunked prefill, and what to do when KV blocks run out
  (queue, preempt, swap, recompute), with a simulation of TTFT, TPOT, ITL and throughput.
step: 9
layer: scheduling
---

# 9. Scheduling

<p class="lede">Paging made memory flexible. The scheduler decides how to spend it. Every
iteration it chooses which requests advance, how much prompt to prefill, and who loses their
blocks when the pool runs dry. These choices move TTFT, inter-token latency and throughput
more than any kernel does.</p>

## Problem

Requests arrive at random times with unknown output lengths. The GPU runs one forward pass at
a time over a batch. How do we form batches so the GPU stays busy, new users get their first
token quickly, and running users don't stutter?

## The naive solution: static batching

Collect up to $B$ requests, prefill them together, decode until **every** one finishes, then
start the next batch. It fails in three ways:

1. **Head-of-line blocking.** A request arriving just after a batch starts waits for the
   whole batch to finish. TTFT includes someone else's longest generation.
2. **Idle slots.** A request that finishes after 20 tokens leaves its slot empty while a
   neighbour generates 800. Utilization is $\sum_r N_r / (B \cdot \max_r N_r)$.
3. **Padding.** Sequences of different lengths are padded to the longest, wasting memory
   and FLOPs.

## Core idea: iteration-level (continuous) batching

Make the scheduling decision **every iteration** instead of every batch (Orca, OSDI 2022). At
iteration $t$:

$$
\mathcal{R}_{t+1} = \big(\mathcal{R}_t \setminus \mathcal{F}_t\big) \cup \mathcal{A}_t,
$$

where $\mathcal{F}_t$ are requests that just emitted EOS or hit a limit, and $\mathcal{A}_t$
are waiting requests admitted because blocks and batch slots are free. A finished request's
slot is refilled on the next iteration. This works because a decode step treats each sequence
independently: sequences at different positions with different context lengths share a
forward pass, with attention handled per sequence over its own block table (step
[7](07-paged-kv.md)).

<div data-widget="batching"></div>

## Admission: memory-aware

Admit request $r$ only if its prompt fits, with headroom for the growth of everyone already
running:

$$
\Big\lceil \tfrac{T_{p,r}}{B_{\text{blk}}} \Big\rceil \;\le\; N_{\text{free}} - W,
$$

where $W$ is a watermark of blocks kept free so running sequences can grow without immediate
preemption. FCFS order is kept: a large request at the head of the queue is not overtaken by
small ones, which would starve it.

## Prefill vs decode inside one iteration

New requests need prefill, which is long and compute-bound. Running requests need one decode
token each, which is short and bandwidth-bound. Three policies:

| Policy | One iteration contains | Effect |
|---|---|---|
| Prefill-first (early vLLM) | only new prompts, while decodes wait | low TTFT, but every running user stalls for the length of the prefill: ITL spikes |
| Decode-first | decodes; prefill only when no decodes are waiting | smooth ITL, but TTFT grows under load |
| **Chunked prefill** (Sarathi-Serve; vLLM V1 default) | all decodes + prompt chunks up to a token budget $\tau$ | bounded iteration time, so bounded ITL; prefill runs alongside decode |

With chunked prefill, each iteration processes at most $\tau$ tokens: $B_{\text{dec}}$ decode
tokens plus $\tau - B_{\text{dec}}$ prompt tokens. A $T_p$-token prompt completes in
$\lceil T_p / (\tau - B_{\text{dec}}) \rceil$ iterations, and every iteration's time is bounded
by roughly $t(\tau)$. Mixing compute-bound prompt chunks into bandwidth-bound decode
iterations also uses tensor cores that decode alone would leave idle (step
[2](02-prefill-and-decode.md)).

## When blocks run out

On-demand growth (step 7) means a decode step can need a block when none is free. The options:

| Strategy | What happens | Cost | Used when |
|---|---|---|---|
| Queue new arrivals | stop admitting | TTFT grows | always, first line of defence |
| Evict cached prefixes | drop refcount-0 cached blocks (LRU) | future cache misses | before touching running requests |
| **Preempt by recompute** | free a victim's blocks; later re-prefill prompt + generated tokens | extra prefill FLOPs | default in vLLM V1 |
| Preempt by swap | copy victim's KV to CPU memory; copy back later | PCIe traffic both ways, CPU memory | long contexts, slow recompute |
| Offload tiers | keep KV in CPU/SSD tier (LMCache, HiCache) | transfer latency on reuse | large reusable prefixes |
| Reduce batch / reject | lower max concurrency, or fail fast with 429 | throughput or availability | overload protection |

**Recompute vs swap.** For a victim with $T$ tokens of context, recompute costs a prefill, about
$2PT/\pi$ (compute-bound). Swap costs two transfers of its KV over a link of bandwidth
$\beta_{\text{link}}$, about $2 T m_{\text{tok}} / \beta_{\text{link}}$. Swap wins when

$$
\frac{2 T m_{\text{tok}}}{\beta_{\text{link}}} < \frac{2 P T}{\pi}
\quad\Longleftrightarrow\quad
\frac{m_{\text{tok}}}{\beta_{\text{link}}} < \frac{P}{\pi}.
$$

$T$ cancels: the choice depends on the model and the hardware, not the length. For Llama-3-8B
on H100 over PCIe Gen5 (64 GB/s): $131072 / 6.4\times10^{10} \approx 2.0\,\mu s$ per token to
swap, against $8\times10^9 / 9.89\times10^{14} \approx 8.1\,\mu s$ per token to recompute at peak
(realistically 12–16 μs). Swap is cheaper on paper. Recompute is still the common default
because it needs no CPU memory, no transfer scheduling, and no synchronization with a copy
engine, and because a preempted request often has a short context. GQA/MLA models with
small $m_{\text{tok}}$ tilt the inequality further toward swap and offload.

Victim choice: the most recently admitted request (LIFO) loses the least work and protects
requests that are nearly done.

## Experiment

<span class="tag theory">theoretical</span> `inference_lab.scheduler.simulate` with Llama-3-8B
on one H100: 4,096 blocks of 16 tokens (65,536 tokens of KV), at most 64 running requests,
token budget $\tau = 512$. Every forward pass is timed with the roofline model, so absolute
values are upper bounds on speed. The comparison between policies is the point. Trace: 200
Poisson arrivals at 20 req/s, prompts 128–2,048, outputs 32–512 (seed 3).

| Policy | Throughput (tok/s) | TTFT mean | TTFT p99 | TPOT mean | ITL p99 | Preemptions |
|---|---|---|---|---|---|---|
| Static (reserves each request's full length) | 2,614 | 4.41 s | 9.08 s | 6.3 ms | 6.9 ms | 0 |
| Continuous, prefill-first | 3,791 | 0.62 s | 1.75 s | 10.4 ms | 54 ms | 26 |
| Continuous + chunked prefill | 4,313 | 0.032 s | 0.079 s | 7.1 ms | 8.7 ms | 0 |

Same arrivals with long prompts (1,024–4,096 tokens):

| Policy | Throughput | TTFT mean | ITL p99 | Preemptions |
|---|---|---|---|---|
| Continuous, prefill-first | 1,981 | 7.24 s | 81 ms | 16 |
| Continuous + chunked prefill | 2,611 | 4.12 s | 9.1 ms | 27 |

How to read them:

- **Static batching has the smoothest decode** (nothing interrupts a batch) and the worst
  TTFT and throughput: requests wait for whole batches, and finished slots idle.
- **Prefill-first continuous batching** fixes TTFT and throughput but puts every prefill
  between some user's tokens. p99 ITL is 8× worse than static. It also admits aggressively,
  then runs out of blocks and preempts.
- **Chunked prefill** wins on every column in the short-prompt run. With long prompts it
  still bounds ITL (9 ms vs 81 ms), but the pool is saturated: TTFT is dominated by queueing,
  and the scheduler preempts. At that load the fix is more memory per token (GQA/MLA, FP8 KV),
  not a smarter scheduler.

## Failure modes

- **Starvation.** Without FCFS or aging, short requests can overtake a long one forever.
- **Preemption cascades.** Preempting a long request and recomputing it can trigger further
  preemptions. Watermarks and admission limits damp this.
- **Unbounded queues.** Past saturation, queueing makes TTFT grow without limit. Load
  shedding (429 responses) is a feature.
- **Prefill-heavy bursts.** A burst of 100K-token prompts can occupy all chunk budget for
  seconds. Some deployments separate prefill entirely (disaggregation, step 12).

## Questions

1. Derive slot utilization for static batching when output lengths are uniform on $[1, N]$
   and the batch size is $B$.
2. Why does chunked prefill raise throughput as well as cutting ITL?
3. Using the swap-vs-recompute inequality, which wins for DeepSeek-V3 ($m_{\text{tok}} = 68.6$
   KiB, 37B active parameters) over PCIe Gen5?
4. Why preempt the most recently admitted request rather than the longest one?

<div class="ie-prov" markdown>

**Provenance**

Source
: Yu et al., "Orca" (OSDI 2022), iteration-level scheduling. Kwon et al. (SOSP 2023),
  preemption by swap and recompute. Agrawal et al., "Sarathi-Serve" (OSDI 2024, arXiv:2403.02310),
  chunked prefill. vLLM V1 scheduler design (github.com/vllm-project/vllm).

Derivation
: The admission inequality and the swap-vs-recompute inequality are written out here.

Implementation
: `inference_lab.scheduler` (static, continuous, chunked; FCFS admission; LIFO recompute
  preemption). Tested: continuous beats static on throughput and TTFT; chunked prefill
  lowers p99 ITL; the system completes under memory pressure.

Experiment
: Theoretical (roofline-timed simulation). Reproduce with
  `notebooks/11_continuous_batching_simulation.ipynb`.

</div>
