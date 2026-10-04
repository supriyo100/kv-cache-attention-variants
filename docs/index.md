---
template: home.html
title: Inference engineering, from tokens to GPU systems
description: >-
  LLM inference engineering explained one step at a time: prefill and decode, the KV cache
  equation, MHA/GQA/MLA, fragmentation, PagedAttention, prefix caching, continuous batching,
  FlashAttention, long context, quantization, speculative decoding and disaggregated serving.
hide:
  - navigation
  - toc
---

## Six layers, six different questions

Most confusion about inference comes from mixing layers. "PagedAttention makes attention
faster" and "FlashAttention saves KV memory" are both wrong for the same reason: each technique
answers one question at one layer. Every page here names the layer it works on.

| Layer | The question it answers | Techniques | Step |
|---|---|---|---|
| Model architecture | How much KV must exist per token? | MHA, MQA, GQA, MLA, sliding window, hybrid SSM | [4](path/04-kv-cache.md), [5](path/05-attention-variants.md) |
| Cache management | Where does that KV physically live? | Paged KV, block tables, block size | [6](path/06-fragmentation.md), [7](path/07-paged-kv.md) |
| Cache sharing | Can requests reuse each other's KV? | Prefix caching, radix trees, copy-on-write | [8](path/08-prefix-sharing.md) |
| Scheduling | Which requests run this iteration? | Continuous batching, chunked prefill, preemption | [9](path/09-scheduling.md) |
| Kernel | How efficiently is attention computed? | FlashAttention, split-KV decoding, fused kernels | [10](path/10-flashattention.md) |
| Hardware and network | What is the GPU actually waiting on? | HBM bandwidth, tensor cores, NVLink, InfiniBand | [2](path/02-prefill-and-decode.md), [12](path/12-scaling-out.md) |

## Work the numbers yourself

The [interactive tools](lab/tools.md) run in your browser: a KV cache calculator for any model
and GPU, a paged block allocator, a prefix-sharing simulator, a batching timeline, and a
speculative decoding payoff curve. The same formulas live in tested Python in
[`src/inference_lab`](https://github.com/supriyo100/kv-cache-attention-variants/tree/main/src/inference_lab),
and the [notebooks](lab/notebooks.md) run every experiment end to end.

## What the field does today

[State of the art](sota/index.md) maps the 2024–2026 techniques (MLA and FlashMLA,
RadixAttention, KIVI, SnapKV, EAGLE-3, chunked prefill, P/D disaggregation with Mooncake and
Dynamo) to the layer they work on, the official repository that implements them, and a notebook
that rebuilds the core idea from scratch.

## How to read a number on this site

Every result carries one of three labels. <span class="tag measured">measured</span> means it was
run, and the hardware is named. <span class="tag theory">theoretical</span> means it follows from
a formula on the page with datasheet inputs. <span class="tag trend">expected trend</span> means
it's the qualitative direction that published work reports, with no number of our own. No
benchmark here is invented. Where a result needs a GPU we don't have, the code is provided and
the result is left blank.
