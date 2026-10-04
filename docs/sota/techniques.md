---
title: "SOTA inference techniques and their most-starred implementations"
description: >-
  State-of-the-art LLM inference techniques mapped to the most-starred open-source implementation
  of each (vLLM, SGLang, nano-vllm, FlashAttention, FlashMLA, LMCache, StreamingLLM, DeepSpec,
  TurboQuant), what each repository does, and an original PyTorch rebuild of the core idea.
---

# Techniques and their implementations

<p class="lede">For each technique, the reference is the most-starred open-source repository that
actually implements it. This page names that repository, the file where the idea lives, and what
our rebuild takes from it. Our rebuilds are original, small PyTorch implementations for learning,
in the tensor conventions of the upstream course. No code is copied from these repositories.</p>

## How the references were chosen

Candidates came from GitHub search (`gh search repos <technique> --sort stars`) plus the known
reference projects for each technique. Keyword search alone is unreliable: it ranks by
description text, and "paged attention" does not surface vLLM. Among repositories that implement
the technique, the one with the most stars is the reference. Stars, licenses and paths were
checked on **2026-10-05**. Repositories without a standard open-source license are linked and
studied only.

| Technique | Most-starred implementation | Stars | License | Runners-up (stars) |
|---|---|---|---|---|
| Paged KV, continuous batching, chunked prefill, hash prefix cache | [vllm-project/vllm](https://github.com/vllm-project/vllm) | 93.2k | Apache-2.0 | SGLang (36.8k), TensorRT-LLM (14.8k), TGI (10.9k), LMDeploy (8.1k) |
| Readable minimal engine (the same ideas in ~1.2k lines) | [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | 15.7k | MIT | LightLLM (4.3k) |
| Radix-tree prefix cache | [sgl-project/sglang](https://github.com/sgl-project/sglang) | 36.8k | Apache-2.0 | — |
| IO-aware attention, split-KV decoding | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | 25.1k | BSD-3-Clause | FlashInfer (6.5k) |
| MLA decode kernel | [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) | 13.0k | MIT | — |
| KV offload, transfer, P/D disaggregation | [LMCache/LMCache](https://github.com/LMCache/LMCache) | 12.0k | Apache-2.0 | Dynamo (8.2k, linked only), Mooncake (6.7k), llm-d (4.7k) |
| KV eviction | [mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm) | 7.3k | MIT | KVCache-Factory (1.4k), NVIDIA kvpress (1.2k), H2O (0.5k), SnapKV (0.3k) |
| Speculative decoding (drafter training and evaluation) | [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec) | 7.2k | MIT | DFlash (6.1k), Medusa (2.8k), EAGLE (2.5k, linked only) |
| KV cache quantization | [0xSero/turboquant](https://github.com/0xSero/turboquant) | 1.8k | GPL-3.0 (linked only) | KVQuant (0.4k), KIVI (0.4k) |
| Expert-parallel communication | [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) | 10.2k | MIT | — |
| Weight quantization | [AutoGPTQ/AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) | 5.1k | MIT | llm-awq (3.6k), GPTQ (2.4k), SmoothQuant (1.7k) |

## S01. Prefix caching: hash-chained blocks and radix trees

**Problem.** Shared system prompts, multi-turn chats and agents repeat long prefixes.
**Reference.** vLLM (`vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/block_pool.py`) for
hash-chained blocks; SGLang (`python/sglang/srt/mem_cache/radix_cache.py`) for the radix tree.
The clearest reading is nano-vllm's `nanovllm/engine/block_manager.py`.
**What we took from it.** The FIFO free list doubles as the LRU (freed blocks keep their hash
until reallocated); hits are verified against stored tokens; the last block is never served
from cache. **Rebuild.** `inference_lab.prefix.HashBlockCache`, `RadixCache`; notebook
`sota_01_radix_prefix_cache`. **Derivation.** [Step 8](../path/08-prefix-sharing.md).

## S02. MLA weight absorption

**Problem.** Up-projecting the latent cache every step costs FLOPs and materializes K and V.
**Reference.** FlashMLA (`flash_mla/`): decode kernels for the absorbed form over a paged
latent cache. **What we took from it.** The decode-time contract: attend over the cached
latent and RoPE key directly, with queries projected into latent space.
**Rebuild.** `inference_lab.mla.absorbed_forward`, built on the upstream course's
`MultiHeadLatentAttention` and tested equal to its `forward()`; notebook
`sota_02_mla_absorption`. **Derivation.** [Step 5](../path/05-attention-variants.md).

## S03. Split-KV decoding and the LSE merge

**Problem.** At long context and small batch, one query per sequence cannot fill the SMs.
**Reference.** flash-attention (`csrc/flash_attn`, split-KV path; `hopper/` for
FlashAttention-3); FlashInfer (`flashinfer/cascade.py`, `flashinfer/decode.py`) for cascade
attention. **What we took from it.** Kernels return the log-sum-exp alongside the output so
partial results merge exactly. **Rebuild.** `inference_lab.flash` (PyTorch, tested against
SDPA); notebook `sota_03_flash_decoding_lse_merge`.
**Derivation.** [Step 10](../path/10-flashattention.md).

## S04. KV quantization: TurboQuant and KIVI

**Problem.** KV dominates memory and bandwidth at long context; outlier channels ruin naive
low-bit quantization. **Reference.** The most-starred implementation is a TurboQuant port with
vLLM integration (0xSero/turboquant, GPL-3.0, so linked only). It implements TurboQuant
(Zandieh et al., arXiv:2504.19874): random rotation, then a fixed near-optimal scalar codebook.
KIVI (jy-yuan/KIVI, `quant/`) is the per-channel alternative. **What we took from it.** The
rotation-plus-fixed-codebook design and its storage format (bits plus a norm per vector).
**Rebuild.** `inference_lab.quant.turboquant`, `kivi`, `lloyd_max_gaussian`; notebook
`sota_04_kv_quant_kivi_turboquant`. **Derivation.** [Step 11](../path/11-long-context.md).

## S05. KV eviction

**Problem.** Bound KV size for very long inputs.
**Reference.** streaming-llm (`streaming_llm/kv_cache.py`, `streaming_llm/enable_streaming_llm.py`):
attention sinks plus a recent window. Multi-policy libraries (KVCache-Factory, NVIDIA kvpress)
package SnapKV, H2O and others for Hugging Face models. **Rebuild.** `inference_lab.eviction`;
notebook `sota_05_kv_eviction`. **Derivation.** [Step 11](../path/11-long-context.md).

## S06. Speculative decoding: block drafters and confidence-scheduled verification

**Problem.** Small-batch decode leaves most FLOPs idle; at high concurrency, verifying long
drafts wastes batch slots. **Reference.** DeepSpec (`deepspec/modeling/{eagle3,dspark,...}`,
`deepspec/eval/`), which trains and evaluates EAGLE-3, DFlash (block diffusion,
arXiv:2602.06036) and DSpark (confidence-scheduled, arXiv:2607.05147) drafters. Engine
integration: vLLM `vllm/v1/spec_decode/eagle.py`. **What we took from it.** The shift from
chain drafting to block drafting, and verifying by confidence under a shared slot budget.
**Rebuild.** `inference_lab.specdec` (acceptance rule with a distribution test, payoff formula,
`confidence_schedule`); notebook `sota_06_speculative_decoding`.
**Derivation.** [Step 12](../path/12-scaling-out.md).

## S07. Chunked prefill

**Problem.** Long prefills stall every running decode.
**Reference.** vLLM V1 scheduler (`vllm/v1/core/sched/scheduler.py`); nano-vllm
`nanovllm/engine/scheduler.py` shows the same token-budget idea in about 70 lines.
**Rebuild.** `inference_lab.scheduler` (`policy="chunked"`); notebook `sota_07_chunked_prefill`.
**Derivation.** [Step 9](../path/09-scheduling.md).

## S08. KV transfer and prefill/decode disaggregation

**Problem.** Prefill and decode interfere and want different resources.
**Reference.** LMCache (`lmcache/`): KV offload to CPU/disk/remote and KV connectors for P/D
disaggregation in vLLM and SGLang. Mooncake (`mooncake-transfer-engine`, `mooncake-store`) and
Dynamo (`lib/llm`, linked only) are the other major systems. **Rebuild.**
`inference_lab.disagg` (transfer vs prefill time model); notebook `sota_08_pd_disaggregation`.
**Derivation.** [Step 12](../path/12-scaling-out.md).
