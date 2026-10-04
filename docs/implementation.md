---
title: "From explanation to implementation: where each idea is explained, and the library component that implements it"
description: >-
  A map from every inference-engineering idea on this site (KV cache, GQA, MLA, RoPE, paged KV,
  prefix caching, scheduling, FlashAttention, KV quantization, eviction, speculative decoding,
  disaggregation) to where it is explained here and to the usable component in PyTorch,
  Hugging Face transformers, vLLM, SGLang, FlashAttention, FlashInfer and kvpress.
hide:
  - toc
---

# From explanation to implementation

<p class="lede">Each idea on this site appears in three places: an explanation (a path step, a
plate, a hand-drawn sheet), a small reference implementation in <code>inference_lab</code> that
you can read in one sitting, and a production component in a library you would actually deploy.
This page connects the three.</p>

**How to read it.** The "Explained" column links to the page that teaches the idea. "Reference"
is this repository's minimal, tested version, written to be read. "In practice" is what to import
or configure. The reference code mirrors the production design so that one reading prepares you
for the other, but it is not a drop-in replacement.

!!! note "Versions and verification"
    PyTorch names were checked against **torch 2.13** and Hugging Face names against
    **transformers 5.18** by importing them. vLLM, SGLang, FlashAttention, FlashInfer,
    FlashMLA and kvpress need a CUDA GPU and were not installed here; their names follow each
    project's documentation and change between releases, so check the version you deploy. None of
    the snippets on this page were executed for this site.

## The map

| Idea | Explained | Reference (`inference_lab`) | In practice |
|---|---|---|---|
| Prefill vs decode, the KV cache | [path 2](path/02-prefill-and-decode.md), [4](path/04-kv-cache.md), [story 1–2](story.md#act-i-one-request) | `kv_math.step_cost`, `roofline_time` | transformers `DynamicCache` / `StaticCache` via `generate()` |
| KV memory math | [path 4](path/04-kv-cache.md), [session 02](sessions/02_kv_cache_memory_math.md) | `kv_math.kv_bytes_per_token`, `max_concurrent_tokens` | `AutoConfig` fields; vLLM's KV pool log line |
| Roofline, arithmetic intensity | [path 3](path/03-attention-cost.md), [story 4](story.md#4-the-roofline) | `kv_math.arithmetic_intensity`, `roofline_time` | Nsight Compute / PyTorch profiler |
| MQA / GQA | [path 5](path/05-attention-variants.md), [sessions 04](sessions/04_mqa.md)–[05](sessions/05_gqa.md) | course `kv_cache_variants.attention` | `scaled_dot_product_attention(enable_gqa=True)`; config `num_key_value_heads` |
| MLA, weight absorption | [path 5](path/05-attention-variants.md#mla-change-what-is-cached), [session 06](sessions/06_mla.md) | `mla.absorbed_forward` | transformers `DeepseekV3Attention`; FlashMLA; vLLM MLA backend |
| RoPE, context extension | [session 07](sessions/07_rope.md), [story 8](story.md#8-position-inside-the-cache-rope) | course `kv_cache_variants.rope` | `apply_rotary_pos_emb`; `rope_scaling` (`yarn`, `llama3`, …) |
| Fragmentation, paged KV | [path 6](path/06-fragmentation.md)–[7](path/07-paged-kv.md) | `fragmentation.simulate`, `paged.PagedKVCache` | vLLM block pool; `flash_attn_with_kvcache(block_table=…)`; FlashInfer paged wrappers |
| Prefix caching, copy-on-write | [path 8](path/08-prefix-sharing.md) | `prefix.HashBlockCache`, `prefix.RadixCache` | vLLM `enable_prefix_caching`; SGLang RadixAttention; transformers prompt-cache reuse |
| Continuous batching, chunked prefill | [path 9](path/09-scheduling.md) | `scheduler.simulate` | vLLM / SGLang schedulers; transformers `generate_batch` |
| FlashAttention, split-KV merge | [path 10](path/10-flashattention.md), [session 08](sessions/08_flashattention.md) | `flash.tiled_attention`, `merge_states` | `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`; `flash_attn`; `flashinfer.merge_state` |
| KV quantization (KIVI, TurboQuant) | [path 11](path/11-long-context.md) | `quant.kivi`, `quant.turboquant` | transformers `QuantizedCache`; vLLM `kv_cache_dtype="fp8"` |
| KV eviction (StreamingLLM, H2O, SnapKV) | [path 11](path/11-long-context.md) | `eviction.streaming`, `h2o`, `snapkv` | NVIDIA kvpress presses; sliding-window attention |
| Speculative decoding | [path 12](path/12-scaling-out.md) | `specdec.speculative_step`, `confidence_schedule` | transformers `assistant_model`, `prompt_lookup_num_tokens`; vLLM `speculative_config` |
| Prefill/decode disaggregation | [path 12](path/12-scaling-out.md) | `disagg.transfer_ratio` | vLLM `kv_transfer_config` (LMCache, NIXL); SGLang PD; NVIDIA Dynamo |
| Tensor parallelism | [path 12](path/12-scaling-out.md) | `kv_math.max_concurrent_tokens(n_gpus=…)` | vLLM `tensor_parallel_size`; transformers `tp_plan="auto"` |

The notebooks in [`notebooks/`](lab/notebooks.md) run the reference versions; the [interactive tools](lab/tools.md)
run the same formulas in the browser.

---

## 1. The KV cache in `generate()`

**Explained:** [path step 2](path/02-prefill-and-decode.md), [step 4](path/04-kv-cache.md),
[story act I](story.md#act-i-one-request).
**Reference:** `inference_lab.kv_math` sizes the cache. The course module
`kv_cache_variants` implements a cache by hand.

In transformers the cache is an object passed through `generate()`. `DynamicCache` grows as
tokens arrive. `StaticCache` preallocates `max_cache_len` slots, which gives `torch.compile`
fixed shapes. That is the contiguous `max_len` reservation from [path step 6](path/06-fragmentation.md),
chosen on purpose for a single sequence.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype="auto")

inputs = tok.apply_chat_template([{"role": "user", "content": "Why is decode memory-bound?"}],
                                 add_generation_prompt=True, return_tensors="pt", return_dict=True)
cache = DynamicCache()
out = model.generate(**inputs, past_key_values=cache, max_new_tokens=64)
# prefill wrote the prompt's K/V; each decode step appended one token per layer
print(cache.get_seq_length())

# fixed shapes for torch.compile: a preallocated StaticCache
out = model.generate(**inputs, cache_implementation="static", max_new_tokens=64)
```

`cache_implementation` also accepts `"offloaded"` (layers stream between CPU and GPU),
`"sliding_window"`, `"hybrid"` (sliding plus full layers, as in Gemma) and `"quantized"` (§ 10).

## 2. Sizing the cache from a real config

**Explained:** [path step 4](path/04-kv-cache.md), [story 2–3](story.md#act-ii-the-budget).
**Reference:** `kv_math.ModelShape`, `kv_bytes_per_token`, `max_concurrent_tokens`.

Everything $M_{KV}$ needs is in the model's `config.json`. Only that small file is downloaded,
not the weights:

```python
from transformers import AutoConfig

c = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")      # gated: needs an HF token
head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
per_token = 2 * c.num_hidden_layers * c.num_key_value_heads * head_dim * 2   # BF16
print(per_token / 1024, "KiB per token")                           # 128.0

# MLA models (DeepSeek-V3) cache kv_lora_rank + qk_rope_head_dim per layer instead
d = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V3")
print(d.num_hidden_layers * (d.kv_lora_rank + d.qk_rope_head_dim) * 2 / 1024, "KiB")   # 68.6
```

In production, vLLM profiles the weights and activations at startup and logs the size of the KV
pool it allocated, in GiB and in tokens. That log line is `max_concurrent_tokens` measured on real
hardware.

## 3. GQA and MQA in PyTorch and transformers

**Explained:** [path step 5](path/05-attention-variants.md), [sessions 04](sessions/04_mqa.md) and
[05](sessions/05_gqa.md), [plate: head wiring](beyond/variants_compared.md#2-fewer-kv-heads).

PyTorch's fused attention accepts fewer K/V heads than query heads directly, with no
`repeat_kv` copy:

```python
import torch
import torch.nn.functional as F

q = torch.randn(1, 32, 1, 128)          # 32 query heads, one new token
k = torch.randn(1, 8, 4096, 128)        # 8 KV heads (GQA-8), 4,096 cached tokens
v = torch.randn(1, 8, 4096, 128)
o = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)   # (1, 32, 1, 128)
```

In transformers the variant is just a config field: `num_key_value_heads` equals
`num_attention_heads` for MHA, a divisor of it for GQA, and 1 for MQA. The eager attention path
expands K/V with `repeat_kv` (see `transformers/models/llama/modeling_llama.py`). The SDPA and
FlashAttention paths pass the grouped heads to the kernel.

## 4. MLA and absorption

**Explained:** [path step 5](path/05-attention-variants.md), [session 06](sessions/06_mla.md),
[plate: MLA](beyond/variants_compared.md#4-compress-what-is-stored),
[sheet § 4](sessions/06_mla.md#sheet-mla@3).
**Reference:** `inference_lab.mla.absorbed_forward` folds `W^UK` into the query and `W^UV` into
`W^O`, and is tested to match the explicit form.

- **transformers** `DeepseekV3Attention` (`models/deepseek_v3/modeling_deepseek_v3.py`)
  readably implements the explicit form: it up-projects K and V from the latent.
- **FlashMLA** (`deepseek-ai/FlashMLA`) is the absorbed decode kernel used in serving:
  `get_mla_metadata(...)` and then `flash_mla_with_kvcache(q, kv_cache, block_table, cache_seqlens, ...)`
  over a paged latent cache.
- **vLLM and SGLang** select an MLA attention backend automatically for DeepSeek-architecture models
  and cache only the 576-number latent per token per layer.

## 5. RoPE and context extension

**Explained:** [session 07](sessions/07_rope.md), [story 8](story.md#8-position-inside-the-cache-rope),
[sheet § 9](sessions/07_rope.md#sheet-rope@8).

- `apply_rotary_pos_emb(q, k, cos, sin)` and `rotate_half` live in each model file, for example
  `modeling_llama.py`. The model's rotary embedding module produces `cos` and `sin`.
- Context extension is configuration, not code. Set `rope_scaling` in the config, with
  `rope_type` set to one of `linear`, `dynamic` (NTK), `yarn`, `longrope`, `llama3` or `proportional`:

```python
from transformers import AutoConfig, AutoModelForCausalLM

cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
cfg.rope_scaling = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", config=cfg)
```

## 6. Paged KV

**Explained:** [path steps 6–7](path/06-fragmentation.md), [story act IV](story.md#act-iv-place-it-without-waste),
[session 10](sessions/10_pagedattention_vllm.md).
**Reference:** `inference_lab.paged.BlockAllocator` and `PagedKVCache` (refcounts and copy-on-write),
and `fragmentation.simulate` (contiguous vs paged).

- **vLLM** owns the allocator. The knobs are `block_size`, `gpu_memory_utilization`,
  `max_model_len` and `max_num_seqs`. To read the source, start with
  `vllm/v1/core/kv_cache_manager.py` and `vllm/v1/core/block_pool.py`.
- **FlashAttention** reads a paged cache directly:

```python
from flash_attn import flash_attn_with_kvcache

# k_cache, v_cache: (num_blocks, block_size, n_kv_heads, head_dim); block_table: (batch, max_blocks) int32
out = flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=seqlens,
                              block_table=block_table, causal=True)
```

- **FlashInfer** provides `BatchDecodeWithPagedKVCacheWrapper` and
  `BatchPrefillWithPagedKVCacheWrapper`. Call `plan(...)` once per batch layout, then
  `run(q, paged_kv_cache)`. Several serving engines use it as their attention backend.

## 7. Prefix caching and sharing

**Explained:** [path step 8](path/08-prefix-sharing.md), [story 13](story.md#13-sharing).
**Reference:** `prefix.HashBlockCache` (hash-chained blocks with an LRU of free cached blocks, as
in nano-vllm and vLLM) and `prefix.RadixCache` (as in SGLang).

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", enable_prefix_caching=True, block_size=16)
system = "You are a support agent for ...\n" * 50          # shared prefix, byte-identical
outs = llm.generate([system + q for q in questions], SamplingParams(max_tokens=128))
```

- **SGLang** turns RadixAttention on by default; `--disable-radix-cache` turns it off.
- **transformers**, for a single process, can prefill a shared prompt once and reuse the cache
  per request. Deep-copy it, because the cache is mutated in place:

```python
import copy
prompt_cache = DynamicCache()
with torch.no_grad():
    prompt_cache = model(**tok(system, return_tensors="pt"), past_key_values=prompt_cache).past_key_values
for q in questions:
    ids = tok(system + q, return_tensors="pt")
    out = model.generate(**ids, past_key_values=copy.deepcopy(prompt_cache), max_new_tokens=64)
```

## 8. Scheduling: continuous batching and chunked prefill

**Explained:** [path step 9](path/09-scheduling.md), [story 14](story.md#14-the-scheduler),
[scheduler sheet](beyond/serving.md#sheet-serving_scheduler@0).
**Reference:** `scheduler.simulate(policy="static" | "continuous" | "chunked")`, with LIFO
recompute preemption.

- **vLLM:** `max_num_seqs` caps the running batch and `max_num_batched_tokens` is the per-step
  token budget that chunked prefill splits a long prompt against. When blocks run out, the
  scheduler preempts and recomputes.
- **SGLang:** `--chunked-prefill-size`, `--max-running-requests`, and `--schedule-policy`
  (for example `lpm`, longest prefix match, which works with the radix cache).
- **transformers** has a built-in continuous-batching loop over a paged cache. Load the model with
  a paged attention implementation (`attn_implementation="paged|sdpa"` or `"paged|flash_attention_2"`)
  and call `model.generate_batch(inputs=[token_ids, ...], generation_config=...)` with lists of token IDs. It is useful for offline
  batches and for reading a compact implementation; use a serving engine for production traffic.

## 9. FlashAttention and split-KV decoding

**Explained:** [path step 10](path/10-flashattention.md), [story 15](story.md#15-the-kernel),
[session 08](sessions/08_flashattention.md).
**Reference:** `flash.tiled_attention` (online softmax), and `flash.merge_states` plus
`split_kv_attention` (the log-sum-exp merge behind Flash-Decoding).

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

- **transformers:** `from_pretrained(..., attn_implementation="sdpa" | "flash_attention_2" | "flex_attention")`.
- **flash-attn:** `flash_attn_func`, `flash_attn_varlen_func` (packed variable-length batches)
  and `flash_attn_with_kvcache`, which also takes `return_softmax_lse=True` for merging partial results.
- **FlashInfer:** `merge_state(v_a, s_a, v_b, s_b)` is exactly `inference_lab.flash.merge_states`.
- **PyTorch FlexAttention** writes masks as Python, for example a sliding window:

```python
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

def sliding(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & (q_idx - kv_idx < 4096)

mask = create_block_mask(sliding, B=None, H=None, Q_LEN=T, KV_LEN=T)
o = flex_attention(q, k, v, block_mask=mask, enable_gqa=True)
```

## 10. KV quantization

**Explained:** [path step 11](path/11-long-context.md), [compare: FP16 vs FP8 vs INT4](compare/index.md).
**Reference:** `quant.kivi` (per-channel keys, per-token values, full-precision recent window) and
`quant.turboquant` (random rotation plus a Lloyd-Max codebook).

```python
# transformers: quantized KV through optimum-quanto or HQQ
out = model.generate(**inputs, max_new_tokens=256,
                     cache_implementation="quantized", cache_config={"backend": "quanto", "nbits": 4})
```

transformers' `QuantizedCache` keeps the most recent `residual_length` tokens (128 by default) in
full precision and quantizes older ones in groups (`q_group_size`, 64). That is the same design as
KIVI's full-precision recent window in `quant.kivi(..., residual=32)`.

- **vLLM:** `kv_cache_dtype="fp8"`, or `"fp8_e4m3"` / `"fp8_e5m2"`. That sets S = 1 byte, which
  halves every block and doubles the tokens per pool.
- **SGLang:** `--kv-cache-dtype fp8_e5m2`.
- KIVI and TurboQuant themselves are research code. Use the authors' repositories (linked from
  [Techniques and implementations](sota/techniques.md)).

## 11. KV eviction and long context

**Explained:** [path step 11](path/11-long-context.md), [plate: state vs cache](beyond/variants_compared.md#8-a-state-instead-of-a-cache).
**Reference:** `eviction.streaming` (attention sinks plus a recent window), `h2o` (heavy
hitters), and `snapkv` (an observation window with pooling).

**NVIDIA kvpress** packages these as *presses* for transformers models:

```python
from transformers import pipeline
from kvpress import SnapKVPress, StreamingLLMPress

pipe = pipeline("kv-press-text-generation", model="Qwen/Qwen2.5-7B-Instruct", device="cuda", dtype="auto")
answer = pipe(long_context, question="What was the 5th number?", press=SnapKVPress(compression_ratio=0.5))["answer"]
```

Models trained with sliding-window layers (Mistral v0.1, Gemma, gpt-oss) cap $T$ by design.
Their config sets `sliding_window`, and the engines allocate their KV accordingly.

## 12. Speculative decoding

**Explained:** [path step 12](path/12-scaling-out.md), [SOTA techniques](sota/techniques.md).
**Reference:** `specdec.speculative_step` (the exact accept/resample rule), `expected_tokens`,
`best_gamma`, and `confidence_schedule`.

```python
# transformers: assisted generation with a small draft model of the same tokenizer family
draft = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype="auto")
out = model.generate(**inputs, assistant_model=draft, max_new_tokens=128)

# no draft model: propose tokens by n-gram lookup in the prompt
out = model.generate(**inputs, prompt_lookup_num_tokens=10, max_new_tokens=128)
```

- **vLLM:** `LLM(model=..., speculative_config={"model": draft_or_eagle_head, "num_speculative_tokens": 4})`.
  Draft models, EAGLE heads and n-gram proposers are all supported.

## 13. Disaggregation and multi-GPU

**Explained:** [path step 12](path/12-scaling-out.md), [sheet: context parallelism](beyond/serving.md#sheet-serving_stack@4).
**Reference:** `disagg.transfer_time` and `transfer_ratio` (when shipping KV beats recomputing it).

- **Tensor parallelism:** vLLM `tensor_parallel_size=N`; transformers
  `from_pretrained(..., tp_plan="auto")` under `torchrun`.
- **Prefill/decode split:** vLLM's `kv_transfer_config`, with connectors such as LMCache and NIXL,
  moves KV between prefill and decode instances. SGLang has a PD-disaggregation mode, and NVIDIA
  Dynamo orchestrates the split across nodes. LMCache also offloads KV to CPU and disk.

---

## Why keep a reference implementation at all?

Production components are optimized for speed. That makes them hard to read and hard to run
without a GPU. The reference versions here are optimized for clarity instead: each runs on a CPU
in seconds, each has a test that pins the behavior the explanation claims, and each follows the
design of the library it stands in for (vLLM's block pool, SGLang's radix tree, FlashAttention's
online softmax). Read the reference to understand the idea, then use the library to ship it.
Sources and licenses for every upstream project are on [Sources and attribution](resources.md).
