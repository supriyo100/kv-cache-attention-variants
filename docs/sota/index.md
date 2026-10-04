---
title: "State of the art in LLM inference (2026): a layer-by-layer map"
description: >-
  A map of current LLM inference techniques by layer (architecture, cache management, sharing,
  scheduling, kernels, distribution), what each one buys, what it costs, and which engines and
  models use it.
---

# State of the art, by layer

<p class="lede">What production LLM serving looks like now, organized by the six layers used
throughout this site. Each entry says what problem a technique solves and what it costs, and
links to the step that derives it and to a notebook that rebuilds its core idea.</p>

!!! note "As of October 2026"
    This field moves monthly. Entries name the paper and the official repository, so you can
    check the current state at the source. Where a claim depends on a specific engine version,
    the engine is named rather than a version number.

## Architecture: what each token leaves behind

| Technique | Solves | Costs | Who uses it | Read |
|---|---|---|---|---|
| GQA | KV bytes per token ↓ $H_q/H_{kv}$ | slight quality loss vs MHA at high $g$ | nearly all dense open models | [step 5](../path/05-attention-variants.md) |
| MLA + absorbed decode | KV ↓ ≈ 57× vs MHA; decode attention near compute-bound | more decode FLOPs; latent replicated under TP | DeepSeek-V2/V3/R1, Kimi K2 and other models built on the DeepSeek-V3 architecture | [step 5](../path/05-attention-variants.md), [notebook S02](../lab/notebooks.md) |
| Local:global attention | KV and attention FLOPs of local layers capped at the window | long-range recall depends on fewer layers | Gemma 2/3, gpt-oss (alternating banded and dense layers) | [step 11](../path/11-long-context.md) |
| Hybrid SSM or linear attention + full attention | constant-size state in most layers | retrieval relies on the few attention layers | Jamba, Nemotron-H, Qwen3-Next-style hybrids | [step 11](../path/11-long-context.md) |
| Mixture of experts | parameters ≫ active FLOPs per token | weights still occupy memory; all-to-all communication | DeepSeek-V3, Qwen3-MoE, gpt-oss, Llama 4 | [step 12](../path/12-scaling-out.md) |
| Multi-token prediction | extra heads double as a speculative draft | training-time change | DeepSeek-V3 | [step 12](../path/12-scaling-out.md) |

## Cache management and sharing: where KV lives and who reuses it

| Technique | Solves | Costs | Implementations | Read |
|---|---|---|---|---|
| Paged KV (block tables) | external fragmentation, reservation waste | indirection in every kernel | vLLM, SGLang, TensorRT-LLM, FlashInfer kernels | [steps 6–7](../path/07-paged-kv.md) |
| Hash-chained prefix caching | duplicate prefixes and their prefill | memory held by cold prefixes; hash bookkeeping | vLLM (on by default in V1) | [step 8](../path/08-prefix-sharing.md) |
| Radix-tree prefix caching | same, at token granularity, multi-turn friendly | tree maintenance on the CPU | SGLang RadixAttention | [step 8](../path/08-prefix-sharing.md), [notebook S01](../lab/notebooks.md) |
| Tiered KV (GPU → CPU → SSD → remote) | capacity beyond HBM; reuse across instances | transfer latency; extra infrastructure | LMCache, SGLang HiCache, Mooncake Store, Dynamo KV block manager | [step 12](../path/12-scaling-out.md) |
| FP8 KV cache | KV capacity and bandwidth ÷2 | small accuracy risk; needs scales | vLLM, SGLang, TensorRT-LLM | [step 11](../path/11-long-context.md) |
| 2–4-bit KV: KIVI (per-channel keys), TurboQuant (random rotation + fixed codebook) | KV ÷4–7 | task-dependent quality; dequant or rotation cost | TurboQuant vLLM port (most-starred), research code | [step 11](../path/11-long-context.md), [notebook S04](../lab/notebooks.md) |
| KV eviction (SnapKV, H2O, StreamingLLM) | bounded KV for very long inputs | lossy; may drop what a later question needs | research code, some engine integrations | [notebook S05](../lab/notebooks.md) |

## Scheduling: who runs each iteration

| Technique | Solves | Costs | Implementations | Read |
|---|---|---|---|---|
| Continuous (iteration-level) batching | head-of-line blocking, idle slots | scheduler complexity | all major engines | [step 9](../path/09-scheduling.md) |
| Chunked prefill | ITL spikes from long prompts | per-iteration token-budget tuning | vLLM V1 default, SGLang, TensorRT-LLM | [step 9](../path/09-scheduling.md), [notebook S07](../lab/notebooks.md) |
| Preemption by recompute or swap | running out of blocks mid-decode | wasted compute or PCIe traffic | vLLM, SGLang | [step 9](../path/09-scheduling.md) |
| Cache-aware routing | prefix hits across replicas | load imbalance vs hit-rate trade-off | SGLang router, Dynamo KV router, llm-d | [step 8](../path/08-prefix-sharing.md) |
| CPU/GPU overlap, persistent batches | scheduler overhead at small step times | engineering complexity | SGLang overlap scheduler, vLLM V1 | [step 12](../path/12-scaling-out.md) |

## Kernels: computing attention and everything around it

| Technique | Solves | Implementations | Read |
|---|---|---|---|
| FlashAttention-2/3 | IO and $O(T^2)$ memory of attention; Hopper async + FP8 | Dao-AILab/flash-attention | [step 10](../path/10-flashattention.md) |
| Split-KV decoding + LSE merge | idle SMs at long-context decode | FlashAttention, FlashInfer | [notebook S03](../lab/notebooks.md) |
| Cascade attention | shared-prefix attention computed once per batch | FlashInfer | [step 10](../path/10-flashattention.md) |
| FlashMLA | MLA decode as a compute-bound kernel | deepseek-ai/FlashMLA | [step 5](../path/05-attention-variants.md) |
| CUDA graphs, fused norms and quantization, fused MoE | launch overhead and extra HBM round trips | all engines | [step 12](../path/12-scaling-out.md) |
| FP8 / FP4 GEMMs | compute-bound prefill, weight bytes | TensorRT-LLM, vLLM, SGLang (Hopper, Blackwell) | [step 12](../path/12-scaling-out.md) |

## Decoding and distribution

| Technique | Solves | Costs | Implementations | Read |
|---|---|---|---|---|
| Speculative decoding: EAGLE-3, MTP, block drafters (DFlash, DSpark) with confidence-scheduled verification | idle FLOPs at small-batch decode; wasted slots at high concurrency | draft training and compute; gains shrink with batch | vLLM, SGLang, TensorRT-LLM; drafters trained with DeepSpec | [step 12](../path/12-scaling-out.md), [notebook S06](../lab/notebooks.md) |
| Prefill/decode disaggregation | phase interference; independent SLOs | KV transfer; two pools to size | Mooncake, Dynamo, vLLM/SGLang KV connectors | [notebook S08](../lab/notebooks.md) |
| Wide expert parallelism | serving huge MoE models efficiently | all-to-all latency (DeepEP) | SGLang, vLLM, TensorRT-LLM | [step 12](../path/12-scaling-out.md) |
| Context parallelism / ring attention | 100K–1M-token prefill | K/V exchange per layer | Megatron-style CP, engine integrations | [step 12](../path/12-scaling-out.md) |

## What to reach for first

| Symptom | Likely layer | First things to try |
|---|---|---|
| Out of KV memory, low batch size | cache / architecture | FP8 KV; prefix caching; check `gpu_memory_utilization`; model with GQA/MLA |
| High TTFT at low load | prefill compute | prefix caching; FP8 prefill; more TP |
| High TTFT only under load | scheduling | more replicas; admission limits; disaggregation |
| ITL spikes | scheduling | chunked prefill with a smaller token budget; disaggregation |
| Low tokens/s at small batch | decode bandwidth | weight quantization; speculative decoding; CUDA graphs |
| Throughput flat as batch grows | KV bandwidth | FP8 KV; GQA/MLA models; shorter contexts per request |

See [Techniques and their implementations](techniques.md) for the most-starred open-source
implementation of each technique, where the idea lives in its code, and what our PyTorch
rebuild takes from it.
