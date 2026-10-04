---
title: "Long-context inference: what breaks at 128K tokens and which fix fits which problem"
description: >-
  Long-context LLM inference broken into six separate problems (KV capacity, KV bandwidth,
  attention compute, scheduling, latency, multi-GPU communication) and mapped to solutions: GQA,
  MLA, KV quantization (KIVI), eviction (StreamingLLM, H2O, SnapKV), sliding window, hybrid SSM
  and context parallelism, with experiments.
step: 11
layer: all layers
---

# 11. Long context

<p class="lede">At 2K tokens most of the techniques on this site are optimizations. At 128K they
are requirements, and they stop being interchangeable. "Long context is expensive" is really six
separate problems. Each technique addresses one or two of them and often makes another
worse.</p>

## Problem: six things break

<span class="tag theory">theoretical</span> Llama-3-8B, BF16, one H100, a single 128K-token
request:

| # | Problem | At 128K | Grows as |
|---|---|---|---|
| 1 | **KV capacity** | 16 GiB per sequence (40 GiB for Llama-3-70B), so 3 fit on one H100 | $T$ |
| 2 | **KV bandwidth** | a decode step reads 17 GB of KV against 16 GB of weights: 9.9 ms vs 4.9 ms at 2K | $T$ per step |
| 3 | **Attention compute** | prefill ≈ 6.7 s at peak; attention is 68% of its FLOPs | $T^2$ |
| 4 | **Scheduling** | one prompt monopolises the chunk budget for hundreds of iterations | $T$ |
| 5 | **Latency** | TTFT is seconds even with an idle GPU | $T^2$ (prefill) |
| 6 | **Multi-GPU communication** | a sequence too large for one GPU's memory or compute must be split | $T$ |

## The naive solutions

"Buy more GPUs" splits the weights, but each GPU still holds and reads KV for its share of
heads (problems 1–2 shrink linearly, problem 6 appears). "Truncate the prompt" solves all six
by discarding the information the user asked about.

## The map: which fix for which problem

| Technique | Layer | Capacity | Bandwidth | Attn compute | Quality risk | Notes |
|---|---|---|---|---|---|---|
| GQA ([step 5](05-attention-variants.md)) | architecture | ↓ $g$× | ↓ $g$× | — | low (pre-trained in) | default in dense models |
| MLA ([step 5](05-attention-variants.md)) | architecture | ↓ ≈ 57× vs MHA | ↓ | ↑ decode FLOPs | low (pre-trained in) | DeepSeek-V2/V3/R1 and models built on them |
| Sliding window / local:global layers | architecture | local layers ↓ to window | ↓ | ↓ to $T \cdot w$ | needs training; long-range recall relies on global layers | Mistral 7B (window), Gemma 2/3 (5 local : 1 global in Gemma 3) |
| Hybrid SSM + attention | architecture | SSM layers have constant state | ↓ | ↓ | needs training | Jamba (1 attention : 7 Mamba), Nemotron-H and other hybrids |
| KV quantization (FP8, KIVI INT2/4) | storage | ↓ 2–5× | ↓ 2–5× | — | low at FP8; medium at 2-bit | FP8 KV is a one-flag change in vLLM/SGLang |
| KV eviction (StreamingLLM, H2O, SnapKV) | cache policy | ↓ to budget | ↓ | ↓ | task-dependent; lossy | best for generation after long prompts |
| Prefix caching ([step 8](08-prefix-sharing.md)) | sharing | shared once | — | skip prefill | none | huge for repeated documents |
| Chunked prefill ([step 9](09-scheduling.md)) | scheduling | — | — | — | none | fixes problem 4, not 5 |
| Context parallelism / ring attention ([step 12](12-scaling-out.md)) | distributed | split across GPUs | split | split | none | fixes 3, 5 and 6 at communication cost |
| Speculative decoding ([step 12](12-scaling-out.md)) | decoding | — | more tokens per KV read | — | none (exact) | helps long-context decode at small batch |

No technique is best overall. GQA and MLA are architectural choices made at pre-training, so
a server inherits whatever the model chose. Quantization and eviction are serving-time
choices with a quality cost. Distribution is a hardware choice with a communication cost.

## KV quantization: why keys and values are grouped differently

Asymmetric $b$-bit quantization of a group $x$ uses a scale $\Delta = (\max x - \min x)/(2^b - 1)$
and stores $q = \operatorname{round}((x - \min x)/\Delta)$. The error per element is at most
$\Delta/2$, so error is set by **the range inside each group**.

Two current answers to the outlier problem:

- **Isolate the outliers: KIVI.** Keys have a few **outlier channels**, large in magnitude for
  every token, while values do not. Group keys per token, and each token's group contains the
  outlier channels, so $\Delta$ is huge for every element. Group keys **per channel** (across
  tokens) instead, and outliers inflate only their own channel's scale. Values are grouped per
  token. The newest tokens stay in full precision until a per-channel group fills.
- **Spread the outliers: TurboQuant** (the most-starred open KV-quantization implementation is
  a TurboQuant port with vLLM integration). Multiply each vector by a random orthogonal matrix
  $R$. Outlier energy spreads over every coordinate, and $y = \sqrt{d}\,R\,x/\|x\|$ has
  nearly i.i.d. $\mathcal N(0,1)$ coordinates, so one fixed Lloyd-Max codebook is near-optimal
  for every vector. Only the norm is stored. The expected relative error is
  $\sqrt{D_b}$: 0.60, 0.34, 0.19, 0.10 at 1–4 bits, regardless of the data. $R$ must be drawn
  independently of the data. Our first version reused the data's random seed for $R$, and the
  "random" rotation stopped making coordinates Gaussian.

<span class="tag measured">measured</span> on synthetic K/V (512 tokens, $d = 128$, 4 outlier
channels at about ±15σ, group 32) with `inference_lab.quant` (PyTorch, CPU, the same computation
the unit tests run):

| Bits | Key error: per-token | Key error: KIVI per-channel | Key error: TurboQuant | Output error: per-token K, V | Output error: KIVI | Output error: TurboQuant | Bytes/elem: group-32 | Bytes/elem: TurboQuant |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.118 | 0.029 | 0.094 | 0.33 | 0.10 | 0.29 | 0.625 | 0.516 |
| 3 | 0.262 | 0.062 | 0.180 | 0.89 | 0.22 | 0.50 | 0.500 | 0.391 |
| 2 | 0.756 | 0.145 | 0.341 | 4.3 | 0.57 | 0.85 | 0.375 | 0.266 |

(Relative Frobenius errors.) Both fixes beat naive per-token grouping by a wide margin. On this
data KIVI's groups of 32 tokens per channel adapt to the outliers and give the lowest error at
equal bits. TurboQuant spends less metadata (one norm per vector instead of a scale and zero per
32 elements), so at equal *bytes* the gap narrows, and its error is guaranteed by the codebook
rather than by the data's structure. At 2 bits every method leaves large error on random data.
Published near-lossless results depend on real attention structure that random data lacks.
Validate low-bit KV on your own task; FP8 is the safe default.

## KV eviction: keep a budget of tokens

Three selection rules, from simplest to most informed:

- **StreamingLLM:** keep the first few tokens ("attention sinks", which absorb surplus
  attention mass) plus a recent window. Content-blind, and stable for unbounded streams.
- **H2O:** keep the recent window plus "heavy hitters", the tokens with the largest
  attention accumulated over past queries.
- **SnapKV:** at the end of prefill, let the last $w$ prompt queries (the observation window)
  vote on which earlier tokens matter, pool votes over neighbours so clustered evidence
  survives, and keep the top-$k$.

<span class="tag measured">measured</span> on a synthetic "needle" context (1,024 tokens, 8
needle tokens that future queries look for, peaked attention), relative error of the attention
output computed over kept tokens only, `inference_lab.eviction`, CPU:

| Budget (of 1,024) | StreamingLLM | H2O | SnapKV |
|---|---|---|---|
| 64 | 1.03 | 0.054 | 0.45 |
| 128 | 1.03 | 0.048 | 0.051 |
| 256 | 2.01 | 0.038 | 0.042 |

StreamingLLM drops the needles regardless of budget, because they are neither early nor
recent. That is the failure mode of content-blind policies on retrieval tasks. H2O and SnapKV
find them. SnapKV's pooling spends part of a small budget on neighbours of needles, which hurts
at 64 and is harmless at 128+. Real models show the same ordering on retrieval benchmarks, but
eviction is lossy by construction: a token dropped now cannot be attended to by a later
question that needs it.

## Sliding windows and hybrids

A sliding window of $w$ tokens caps a layer's cache at $w$ tokens and its attention compute at
$T \cdot w$. Interleaving 5 local layers ($w = 1024$) per global layer, as Gemma 3 does, cuts
the average cache per layer at 128K to $(5 \cdot 1024 + 131072)/6 \approx 22.7$K tokens, 17% of
a full-attention model. Information flows across long distances through the global layers
only. State-space layers (Mamba) replace the growing cache with a fixed-size state, and hybrids
keep a few attention layers for precise retrieval.

## Failure modes

- **"Supports 128K" vs "uses 128K."** A model can accept a long context and still fail to
  retrieve from its middle. Serving optimizations cannot fix training.
- **Eviction before the question.** Prompt-time eviction (SnapKV) uses the end of the prompt
  to decide. If the user's question comes in a *later* turn, the needed tokens may already be
  gone.
- **Quantization error compounds with length.** Small per-token errors in K shift softmax
  weights across more tokens. Validate at the context lengths you serve, not at 2K.

## Questions

1. Which of the six problems does FP8 KV quantization *not* help?
2. Why can StreamingLLM generate stably for millions of tokens but fail a needle-in-a-haystack
   test?
3. For a 1M-token context on Llama-3-70B, compute the per-sequence KV with BF16 GQA, with FP8,
   and with a 5:1 local:global design (window 4096).

<div class="ie-prov" markdown>

**Provenance**

Source
: KIVI (Liu et al., ICML 2024, arXiv:2402.02750; github.com/jy-yuan/KIVI). TurboQuant (Zandieh
  et al., arXiv:2504.19874; implementation github.com/0xSero/turboquant, GPL-3.0, linked only). StreamingLLM
  (Xiao et al., ICLR 2024, arXiv:2309.17453). H2O (Zhang et al., NeurIPS 2023,
  arXiv:2306.14048). SnapKV (Li et al., arXiv:2404.14469). Gemma 3 technical report (2025).
  Jamba (arXiv:2403.19887). Mistral 7B (arXiv:2310.06825).

Derivation
: The six-problem decomposition and the technique-to-problem map are this site's.

Implementation
: `inference_lab.quant` and `inference_lab.eviction` (PyTorch) reduce each method to its
  selection, grouping or rotation rule. They are not the authors' code.

Experiment
: Synthetic, CPU, deterministic. Notebooks `sota_04_kv_quant_kivi_turboquant` and
  `sota_05_kv_eviction`. Results on synthetic data do not transfer quantitatively to real models.

</div>
