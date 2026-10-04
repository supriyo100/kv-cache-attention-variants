---
title: "LLM inference comparisons: MHA vs GQA vs MLA, paged vs contiguous, FP16 vs FP8 vs INT4 and more"
description: >-
  Side-by-side comparisons of LLM inference techniques: attention variants, KV allocation,
  fragmentation types, attention kernels, prefill vs decode, batching policies, number formats,
  bound regimes, parallelism schemes, KV compression approaches and prefix caching.
---

# Comparisons

<p class="lede">Eleven head-to-head comparisons. Each row says what problem a technique solves,
what it costs in memory, latency, throughput, complexity and quality, and when to choose it.
Every row links to the step that derives it.</p>

## MHA vs MQA vs GQA vs MLA

Shapes: $H_q = 128$, $d_h = 128$, BF16, per token per layer. Derivation in [step 5](../path/05-attention-variants.md).

| | MHA | MQA | GQA-8 | MLA |
|---|---|---|---|---|
| Problem solved | — (baseline) | KV size | KV size, keeping quality | KV size, keeping per-head keys |
| Cached elements | 32,768 | 256 | 2,048 | 576 |
| Decode attention intensity | 1 | 128 | 16 | ≈ 242 |
| Decode latency at long context | worst | best bytes | good | good; more FLOPs |
| Implementation complexity | low | low | low | high (absorption, decoupled RoPE, special kernels) |
| Quality | baseline | measurable loss | near MHA | near or above MHA (DeepSeek-V2 report) |
| Choose when | small models or short context | extreme memory limits | default for dense models | designing a model for long-context serving |

## Contiguous vs paged KV allocation

[Steps 6–7](../path/07-paged-kv.md).

| | Contiguous (reserve prompt + max_tokens) | Paged (blocks on demand) |
|---|---|---|
| Problem solved | simple addressing | fragmentation and reservation waste |
| Memory utilization (our simulation) | 65.5% | 99.4% |
| External fragmentation | yes (169 of 300 rejected with free memory) | none at block granularity |
| Internal fragmentation | none | < $B$ slots per sequence |
| Kernel complexity | stride addressing | block-table gather |
| Running out mid-request | impossible (reserved) | possible, so preemption is needed |
| Choose when | single-request, fixed-length workloads | any multi-tenant server |

## External vs internal fragmentation

[Step 6](../path/06-fragmentation.md).

| | External | Internal |
|---|---|---|
| Definition | free memory exists but no contiguous hole fits | allocated slots inside a block hold no token |
| Cause | variable-size allocations, out-of-order frees | rounding up to whole blocks |
| Size | unbounded; can block admission entirely | $< B$ per sequence, mean $(B-1)/2$ |
| Fixed by paging? | yes | no, only bounded |
| Not to be confused with | duplication (same prefix stored many times), fixed by sharing | reservation waste (unused `max_tokens` slots) |

## Naive attention vs FlashAttention

[Step 10](../path/10-flashattention.md).

| | Naive (three kernels) | FlashAttention |
|---|---|---|
| Problem solved | — | HBM traffic and $O(T^2)$ memory |
| Extra memory | $O(T^2)$ per head ($S$, $P$) | $O(T)$ (LSE per row) |
| HBM traffic | $4Td + 4T^2$ | $\Theta(T^2 d^2 / M)$ |
| Exact? | yes | yes |
| Reduces KV cache? | no | **no** |
| Choose when | never in production | always |

## Prefill vs decode

[Step 2](../path/02-prefill-and-decode.md).

| | Prefill | Decode |
|---|---|---|
| Tokens per step per sequence | $T_p$ (all at once) | 1 |
| Arithmetic intensity (BF16) | $\approx T_p$ | $\approx B$, capped by $2P/(T m_{\text{tok}})$ |
| Bound | compute (above ≈ 300 tokens on H100) | HBM bandwidth |
| Sets | TTFT | TPOT / ITL |
| Helped by | FP8 compute, prefix caching, TP, CP | batching, weight and KV quantization, GQA/MLA, speculative decoding |

## Static vs continuous batching

[Step 9](../path/09-scheduling.md). Our roofline-timed simulation, 200 requests at 20 req/s.

| | Static | Continuous (prefill-first) | Continuous + chunked prefill |
|---|---|---|---|
| Throughput | 2,614 tok/s | 3,791 | 4,313 |
| TTFT mean | 4.41 s | 0.62 s | 0.032 s |
| ITL p99 | 6.9 ms | 54 ms | 8.7 ms |
| Complexity | low | medium | medium-high |
| Choose when | offline jobs with uniform lengths | — | interactive serving |

## FP16 vs BF16 vs FP8 vs INT8 vs INT4

[Step 12](../path/12-scaling-out.md).

| | FP16 | BF16 | FP8 (E4M3) | INT8 (W8A8) | INT4 (weight-only) |
|---|---|---|---|---|---|
| Bytes / weight | 2 | 2 | 1 | 1 | ≈ 0.53 (with group scales) |
| Range | narrow (overflow risk) | FP32 exponent range | per-tensor/per-channel scales | scales + outlier handling | group scales |
| Speeds up decode | — | — | ≈ 2× bytes | ≈ 2× bytes | ≈ 3.5× bytes |
| Speeds up prefill | — | — | yes (FP8 tensor cores) | yes (INT8 tensor cores) | little (dequantization) |
| Quality risk | low | low | low | low-medium | medium; method-dependent (AWQ, GPTQ) |
| Choose when | legacy | default | Hopper/Blackwell serving | Ampere serving | memory-limited, small-batch decode |

## Compute-bound vs memory-bound

[Step 2](../path/02-prefill-and-decode.md).

| | Compute-bound | Memory-bound |
|---|---|---|
| Condition | intensity $> \pi/\beta$ (≈ 295 on H100) | intensity $< \pi/\beta$ |
| Typical work | prefill, large-batch GEMMs | decode, attention over long KV |
| What helps | lower-precision tensor cores, better kernels | fewer bytes: quantization, GQA/MLA, batching (weights only) |
| What doesn't | fewer bytes | more FLOPs or faster tensor cores |
| Right metric | achieved TFLOP/s | achieved TB/s |

## Tensor vs pipeline (vs expert vs context) parallelism

[Step 12](../path/12-scaling-out.md).

| | Tensor | Pipeline | Expert | Context |
|---|---|---|---|---|
| Splits | matrices (heads) | layers | MoE experts | sequence tokens |
| Communication | 2 all-reduces / layer | activations between stages | all-to-all per MoE layer | K/V or partial outputs per layer |
| Latency per token | ↓ | ↑ (pipeline depth) | depends on routing | ↓ for long prefill |
| Network need | NVLink | modest | high (all-to-all) | high |
| Choose when | model doesn't fit, or latency matters, inside one node | across nodes | large MoE | very long single sequences |

## KV quantization vs eviction vs architectural compression

[Steps 5](../path/05-attention-variants.md) and [11](../path/11-long-context.md).

| | Quantization (FP8, KIVI) | Eviction (SnapKV, H2O) | Architecture (GQA, MLA) |
|---|---|---|---|
| Keeps every token? | yes, at lower precision | no | yes |
| Saving | 2–5× | up to (context / budget) | 4–57× |
| When decided | serving time | serving time | pre-training |
| Quality risk | low (FP8) to medium (2-bit) | task-dependent, irreversible | trained in |
| Composes with the others? | yes | yes | yes |

## Prefix caching on vs off

[Step 8](../path/08-prefix-sharing.md). Our simulation: 50 users, 2,000-token shared prompt.

| | Off | On |
|---|---|---|
| KV blocks for 50 users (prompt + question) | 6,900 | 775 |
| Prompt tokens prefilled | 110,000 | 12,000 |
| Effect on TTFT | — | lower in proportion to the hit rate |
| Effect on TPOT | — | none |
| Risks | — | memory held by cold prefixes; cross-tenant timing leaks; template drift kills hits |
