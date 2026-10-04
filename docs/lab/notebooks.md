---
title: Notebooks
description: >-
  Original PyTorch notebooks for LLM inference engineering: KV cache sizing, fragmentation, paged
  allocation, batching, prefill vs decode, FlashAttention, and eight state-of-the-art techniques
  rebuilt from their most-starred implementations.
---

# Notebooks

<p class="lede">Original notebooks, separate from the upstream course's
<code>teaching_notebooks/</code>. Each one follows the same sections: objective, theory,
equations, diagram, implementation, experiment, results, interpretation, limitations,
conclusion. They are thin: the code they call lives in <code>inference_lab</code>, in PyTorch
and in the course's tensor conventions, and is covered by the test suite.</p>

!!! note "Published without outputs"
    The notebooks are committed unexecuted, so you see code, not someone else's results. Run them
    locally (`uv sync --extra dev`, then `uv run jupyter lab notebooks/`) or open them in Colab
    from the links below. The first cell installs `inference_lab` if it is missing. Everything
    runs on CPU. The heaviest cells are optional and say so.

## Core notebooks

| Notebook | What it shows | Step |
|---|---|---|
| 05 KV cache memory calculator | KV bytes for real models, concurrency vs context and dtype, variants compared | [4](../path/04-kv-cache.md) |
| 07 External and internal fragmentation | contiguous vs paged allocation on one trace; internal waste vs block size | [6](../path/06-fragmentation.md) |
| 08 Paged KV allocator | block tables, address translation, fork, copy-on-write, pool exhaustion | [7](../path/07-paged-kv.md) |
| 11 Continuous batching | static vs continuous vs chunked prefill; load sweep of TTFT and p99 ITL | [9](../path/09-scheduling.md) |
| 12 Prefill vs decode | arithmetic intensity, roofline step times, the KV ceiling on batching | [2](../path/02-prefill-and-decode.md) |
| 13 FlashAttention IO model | online softmax checked against SDPA; HBM traffic and memory of naive vs tiled | [10](../path/10-flashattention.md) |

## State-of-the-art notebooks

Each rebuilds the core idea of a technique whose most-starred implementation is listed on
[Techniques and implementations](../sota/techniques.md).

| Notebook | Technique | Reference implementation |
|---|---|---|
| S01 Radix vs hash prefix cache | RadixAttention and hash-chained block caching, eviction under pressure | SGLang, vLLM, nano-vllm |
| S02 MLA absorption | absorbed decode on the course's `MultiHeadLatentAttention`, checked equal to its `forward()` | FlashMLA |
| S03 Split-KV and LSE merge | FlashDecoding and cascade attention, checked against SDPA | flash-attention, FlashInfer |
| S04 KV quantization | per-token vs KIVI vs TurboQuant rotation with a Lloyd-Max codebook | TurboQuant port (linked), KIVI |
| S05 KV eviction | StreamingLLM vs H2O vs SnapKV at equal budgets | streaming-llm, kvpress |
| S06 Speculative decoding | exactness, payoff, confidence-scheduled verification of block drafts | DeepSpec (EAGLE-3, DFlash, DSpark) |
| S07 Chunked prefill | the token budget's U-shaped effect on TTFT and its linear effect on ITL | vLLM V1, nano-vllm |
| S08 P/D disaggregation | KV transfer time against prefill time across links and models | LMCache, Mooncake |

## Open a notebook

| Notebook | GitHub | Colab |
|---|---|---|
| 05_kv_cache_memory_calculator | [view](https://github.com/supriyo100/kv-cache-attention-variants/blob/main/notebooks/05_kv_cache_memory_calculator.ipynb) | [open](https://colab.research.google.com/github/supriyo100/kv-cache-attention-variants/blob/main/notebooks/05_kv_cache_memory_calculator.ipynb) |
| 07_external_internal_fragmentation | [view](https://github.com/supriyo100/kv-cache-attention-variants/blob/main/notebooks/07_external_internal_fragmentation.ipynb) | [open](https://colab.research.google.com/github/supriyo100/kv-cache-attention-variants/blob/main/notebooks/07_external_internal_fragmentation.ipynb) |
| 08_paged_kv_allocator_simulation | [view](https://github.com/supriyo100/kv-cache-attention-variants/blob/main/notebooks/08_paged_kv_allocator_simulation.ipynb) | [open](https://colab.research.google.com/github/supriyo100/kv-cache-attention-variants/blob/main/notebooks/08_paged_kv_allocator_simulation.ipynb) |
| 11_continuous_batching_simulation | [view](https://github.com/supriyo100/kv-cache-attention-variants/blob/main/notebooks/11_continuous_batching_simulation.ipynb) | [open](https://colab.research.google.com/github/supriyo100/kv-cache-attention-variants/blob/main/notebooks/11_continuous_batching_simulation.ipynb) |
| 12_prefill_vs_decode | [view](https://github.com/supriyo100/kv-cache-attention-variants/blob/main/notebooks/12_prefill_vs_decode.ipynb) | [open](https://colab.research.google.com/github/supriyo100/kv-cache-attention-variants/blob/main/notebooks/12_prefill_vs_decode.ipynb) |
| 13_flash_attention_io_model | [view](https://github.com/supriyo100/kv-cache-attention-variants/blob/main/notebooks/13_flash_attention_io_model.ipynb) | [open](https://colab.research.google.com/github/supriyo100/kv-cache-attention-variants/blob/main/notebooks/13_flash_attention_io_model.ipynb) |
| sota_01_radix_prefix_cache | [view](https://github.com/supriyo100/kv-cache-attention-variants/blob/main/notebooks/sota_01_radix_prefix_cache.ipynb) | [open](https://colab.research.google.com/github/supriyo100/kv-cache-attention-variants/blob/main/notebooks/sota_01_radix_prefix_cache.ipynb) |
| sota_02_mla_absorption | [view](https://github.com/supriyo100/kv-cache-attention-variants/blob/main/notebooks/sota_02_mla_absorption.ipynb) | [open](https://colab.research.google.com/github/supriyo100/kv-cache-attention-variants/blob/main/notebooks/sota_02_mla_absorption.ipynb) |
| sota_03_flash_decoding_lse_merge | [view](https://github.com/supriyo100/kv-cache-attention-variants/blob/main/notebooks/sota_03_flash_decoding_lse_merge.ipynb) | [open](https://colab.research.google.com/github/supriyo100/kv-cache-attention-variants/blob/main/notebooks/sota_03_flash_decoding_lse_merge.ipynb) |
| sota_04_kv_quant_kivi_turboquant | [view](https://github.com/supriyo100/kv-cache-attention-variants/blob/main/notebooks/sota_04_kv_quant_kivi_turboquant.ipynb) | [open](https://colab.research.google.com/github/supriyo100/kv-cache-attention-variants/blob/main/notebooks/sota_04_kv_quant_kivi_turboquant.ipynb) |
| sota_05_kv_eviction | [view](https://github.com/supriyo100/kv-cache-attention-variants/blob/main/notebooks/sota_05_kv_eviction.ipynb) | [open](https://colab.research.google.com/github/supriyo100/kv-cache-attention-variants/blob/main/notebooks/sota_05_kv_eviction.ipynb) |
| sota_06_speculative_decoding | [view](https://github.com/supriyo100/kv-cache-attention-variants/blob/main/notebooks/sota_06_speculative_decoding.ipynb) | [open](https://colab.research.google.com/github/supriyo100/kv-cache-attention-variants/blob/main/notebooks/sota_06_speculative_decoding.ipynb) |
| sota_07_chunked_prefill | [view](https://github.com/supriyo100/kv-cache-attention-variants/blob/main/notebooks/sota_07_chunked_prefill.ipynb) | [open](https://colab.research.google.com/github/supriyo100/kv-cache-attention-variants/blob/main/notebooks/sota_07_chunked_prefill.ipynb) |
| sota_08_pd_disaggregation | [view](https://github.com/supriyo100/kv-cache-attention-variants/blob/main/notebooks/sota_08_pd_disaggregation.ipynb) | [open](https://colab.research.google.com/github/supriyo100/kv-cache-attention-variants/blob/main/notebooks/sota_08_pd_disaggregation.ipynb) |

## Planned

01 token generation, 02 attention memory math, 03 MHA/MQA/GQA, 06 KV growth, 15 long-context
memory, 16 inference benchmarking (GPU), 17 GPU memory bandwidth (GPU). The plan is tracked in
[`IMPLEMENTATION_PLAN.md`](https://github.com/supriyo100/kv-cache-attention-variants/blob/main/IMPLEMENTATION_PLAN.md).
