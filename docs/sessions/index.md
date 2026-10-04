---
title: About the upstream course
description: >-
  The course sessions this repository forks: naive decoding, KV cache memory math, MHA, MQA,
  GQA, MLA, RoPE, FlashAttention, PyTorch SDPA and PagedAttention, by the upstream author.
---

# The upstream course

!!! info "Upstream material"
    The sessions in this section, the `src/kv_cache_variants` package, `teaching_notebooks/` and
    the D01–D12 diagrams come from
    [sourangshupal/kv-cache-attention-variants](https://github.com/sourangshupal/kv-cache-attention-variants).
    This fork adds Excalidraw drawings next to the text they explain. The original path of this
    site starts at [The path of one request](../path/index.md).

The course builds the story from naive decoding to PagedAttention, one idea at a time, with
every attention variant implemented as a PyTorch module.

| # | Session | What you leave knowing | Source module |
|---|---|---|---|
| 00 | [Foundations](00_foundations.md) | Tokens, embeddings, self-attention, causal masking, autoregressive generation | — |
| 01 | [Naive decoding](01_naive_decoding.md) | Why decoding without a cache is $O(n^2)$ | `naive_decode.py` |
| 02 | [KV cache memory math](02_kv_cache_memory_math.md) | The `kv_cache_bytes` formula | `memory_calc.py` |
| 03 | [MHA recap](03_mha_recap.md) | Multi-head attention, prefill vs incremental decode | `attention/mha.py` |
| 04 | [MQA](04_mqa.md) | One shared K/V head | `attention/mqa.py` |
| 05 | [GQA](05_gqa.md) | The tunable middle ground | `attention/gqa.py` |
| 06 | [MLA](06_mla.md) | Low-rank latent K/V with decoupled RoPE | `attention/mla.py` |
| 07 | [RoPE](07_rope.md) | Rotary encoding and context extension | `rope.py` |
| 08 | [FlashAttention](08_flashattention.md) | IO-awareness and SDPA backends | `sdpa_backends.py` |
| 09 | [PyTorch SDPA](09_pytorch_sdpa.md) | Migrating hand-written attention to the fused op | — |
| 10 | [PagedAttention / vLLM](10_pagedattention_vllm.md) | Block-table KV allocation | `memory_calc.py` |

[D01 · Curriculum dependency: which session builds on which](../assets/diagrams/D01_curriculum_dependency.html){ .diagram }

## Where this site goes further

| Course session | Continued in |
|---|---|
| 02 Memory math | [Step 4](../path/04-kv-cache.md): capacity on real GPUs, MLA counted correctly, a calculator |
| 06 MLA | [Step 5](../path/05-attention-variants.md): the absorption derivation, run on this course's own module |
| 08 FlashAttention | [Step 10](../path/10-flashattention.md): online-softmax proof, IO model, split-KV merge |
| 10 PagedAttention | Steps [6](../path/06-fragmentation.md)–[9](../path/09-scheduling.md): fragmentation measured, prefix sharing, scheduling and preemption |

One correction carried forward: `memory_calc.kv_cache_bytes` suggests modelling MLA as
`num_kv_heads=1, head_dim=latent_dim`. Its leading factor of 2 assumes separate K and V, so
that overstates MLA's cache by about 2×. MLA stores one latent plus a small RoPE key per token
per layer ([step 4](../path/04-kv-cache.md)).
