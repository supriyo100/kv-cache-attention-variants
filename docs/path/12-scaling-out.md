---
title: "Scaling out: quantization, speculative decoding, parallelism and P/D disaggregation"
description: >-
  The last mile of LLM inference engineering: weight and activation quantization (FP8, INT8, INT4,
  AWQ, GPTQ, SmoothQuant, FP4), speculative decoding with its exactness guarantee and speedup
  formula, tensor/pipeline/expert/context parallelism and their communication, and prefill/decode
  disaggregation with KV transfer costs.
step: 12
layer: optimization and distribution
---

# 12. Scaling out

<p class="lede">By now the request is well served on one GPU. This step covers the four levers
left when one GPU is not enough, or not fast enough: fewer bits per number, more tokens per
forward pass, more GPUs per model, and separate GPUs for the two phases. Each lever buys
something specific from step 2's roofline and pays for it somewhere else.</p>

## 12.1 Quantization: fewer bytes per number

**Problem.** Decode streams every weight once per step (step [2](02-prefill-and-decode.md)).
Llama-3-70B in BF16 is 141 GB: two H100s just to hold it, and 42 ms per decode step to read it
on one GPU's bandwidth.

**Core idea.** Store numbers in fewer bits. A step that moves half the bytes takes about half
the time when bandwidth-bound. With low-precision tensor cores (FP8, FP4), compute-bound
prefill speeds up too.

| Format | Bits | 70B weights | What it speeds up | Typical quality impact | Notes |
|---|---|---|---|---|---|
| BF16 / FP16 | 16 | 141 GB | baseline | — | BF16 has FP32's exponent range, so it is the training default |
| FP8 (E4M3) W8A8 | 8 | 71 GB | decode (bytes) and prefill (2× tensor-core FLOPs on Hopper) | small with per-tensor or per-channel scales | first-class on H100/H200/B200 |
| INT8 W8A8 (SmoothQuant) | 8 | 71 GB | both | small if activation outliers are migrated into weights | Ampere-friendly |
| INT4 weight-only (GPTQ, AWQ) | 4 | ≈ 37 GB with group scales | decode (bytes); prefill dequantizes, so little gain | small to moderate; method matters | GPTQ: second-order error compensation. AWQ: protect salient channels found from activations |
| FP4 (NVFP4, MXFP4) | 4 | ≈ 37 GB | both on Blackwell FP4 tensor cores | needs block scaling; some models are released natively in FP4 | microscaling: one scale per small block |

$$
t_{\text{decode}} \approx \frac{s_w P + B T m_{\text{tok}}}{\beta}
$$

Weight quantization shrinks the first term, KV quantization (step 11) the second. Which
matters depends on $B T$.

**Trade-offs.** Lower bits add dequantization work, which hurts compute-bound prefill if no
native low-bit tensor core exists. Outliers (a few large activation channels) are the main
source of error, and every method above is a different way of handling them. Always evaluate
on the target task. Perplexity hides failures on reasoning and long-context retrieval.

## 12.2 Speculative decoding: more tokens per weight read

**Problem.** At small batch, a decode step uses about 1% of the GPU's FLOPs (step 2). Each
weight read produces one token.

**Core idea.** A cheap draft proposes $\gamma$ tokens. The target model scores all $\gamma$ in
**one** forward pass, which costs about the same as scoring one, because the pass is
bandwidth-bound. Accept a prefix of the draft, and always gain at least one token.

**The acceptance rule.** Draft token $x \sim q$ is accepted with probability
$\min\!\big(1, p(x)/q(x)\big)$. On the first rejection, sample a replacement from
$\operatorname{norm}\big(\max(0, p - q)\big)$. Then the emitted token is distributed **exactly**
as $p$:

$$
\Pr[\text{emit } x] = \underbrace{q(x)\min\!\Big(1,\tfrac{p(x)}{q(x)}\Big)}_{\min(p(x),\,q(x))} + \Big(1 - \sum_y \min(p(y), q(y))\Big)\frac{\max(0, p(x) - q(x))}{\sum_y \max(0, p(y)-q(y))} = p(x),
$$

since $\sum_y \max(0, p-q) = 1 - \sum_y \min(p, q)$. Speculation changes speed, never the
output distribution. The per-position acceptance rate is $\alpha = \sum_x \min(p, q) = 1 - \mathrm{TV}(p, q)$.

**The payoff.** With i.i.d. acceptance $\alpha$, tokens per target pass and speedup are

$$
\mathbb{E}[\text{tokens}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha},\qquad
\text{speedup} = \frac{\mathbb{E}[\text{tokens}]}{\gamma c + 1},
$$

where $c$ is the draft's cost relative to one target pass.

<span class="tag theory">theoretical</span> from `inference_lab.specdec` (formula verified by
Monte Carlo in the tests):

| $\alpha$ | $\mathbb{E}$, $\gamma=2$ | $\gamma=4$ | $\gamma=8$ | Best $\gamma$ at $c = 0.05$ | Speedup |
|---|---|---|---|---|---|
| 0.6 | 1.96 | 2.31 | 2.47 | 4 | 1.92× |
| 0.7 | 2.19 | 2.77 | 3.20 | 6 | 2.35× |
| 0.8 | 2.44 | 3.36 | 4.33 | 8 | 3.09× |
| 0.9 | 2.71 | 4.10 | 6.13 | 13 | 4.67× |

<div data-widget="specdec"></div>

**State of the art.** Separate small draft models gave way to drafts built into the target:
Medusa adds extra decoding heads, and EAGLE-1/2/3 drafts from the target's own hidden features
and verifies a **tree** of candidates in one pass. DeepSeek-V3 was trained with multi-token
prediction, which doubles as a draft. The newest drafters are *parallel*: they propose a whole
block in one draft pass. DFlash (arXiv:2602.06036) uses a small block-diffusion model. DSpark
(DeepSeek, arXiv:2607.05147) adds dependencies between drafted tokens and schedules how much of
each block to verify by confidence. The most-starred codebase for training and evaluating
drafters, [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec) (MIT), implements
EAGLE-3, DFlash and DSpark side by side.

**Confidence-scheduled verification.** A block drafter's acceptance decays along the block, and
at high concurrency every verified token takes a batch slot another request could use.
Verifying draft $j$ of a request is worth $\prod_{i \le j} \alpha_i$ expected tokens, a value
that falls along the block. So with a shared slot budget, the optimal policy verifies the
drafts with the largest cumulative confidence across all requests: long verifications for
confident requests, short ones for unsure requests.

<span class="tag theory">theoretical</span> `specdec.confidence_schedule` vs a uniform verify
length, 64 requests, 8-token drafts whose acceptance varies by request (0.5–0.95) and decays
20% along the block:

| Slots per request | Fixed length: expected tokens | Confidence-scheduled | Gain |
|---|---|---|---|
| 2 | 108.5 | 112.8 | +4.0% |
| 3 | 139.6 | 147.9 | +5.9% |
| 4 | 161.5 | 171.4 | +6.1% |
| 6 | 187.8 | 196.4 | +4.6% |
| 9 (verify everything) | 203.8 | 203.8 | 0 |

The gain exists only when slots are scarce, which is the high-concurrency regime where plain
speculative decoding stops paying off.

**Trade-offs.** The gain shrinks as batch grows. At large batch the step is no longer
bandwidth-bound, so verifying $\gamma$ extra tokens per sequence costs real FLOPs. Rejected
drafts also burn compute, and the draft's KV cache needs memory.

## 12.3 Parallelism: more GPUs per model

| Scheme | Splits | Communication per layer | Good for | Cost |
|---|---|---|---|---|
| **Tensor (TP)** | each matrix by heads / columns | 2 all-reduces of activations ($B t d$ each) | latency; models too big for one GPU | all-reduce on every layer; needs NVLink |
| **Pipeline (PP)** | layers into stages | send activations stage → stage | very large models; across nodes | bubbles; higher per-token latency |
| **Data (DP)** | replicas | none (independent) | throughput | memory: full copy each |
| **Expert (EP)** | MoE experts across GPUs | all-to-all dispatch + combine of tokens | large MoE (DeepSeek-V3, Qwen3-MoE) | all-to-all latency; load imbalance |
| **Context / sequence (CP)** | one sequence's tokens | exchange K/V blocks (ring) or partial outputs (LSE merge) | 100K–1M-token prefill | K/V transfer per layer |

A ring all-reduce of $S$ bytes over $N$ GPUs moves $2\frac{N-1}{N} S$ bytes per GPU. In
prefill, $S$ is large and the all-reduce is bandwidth-bound. In decode, $S = B \cdot d \cdot s$ is
small (1 MiB for $B=64$, $d=8192$) and the all-reduce is **latency-bound**: a few microseconds
each, times $2L$ per step. That is why TP beyond one NVLink domain rarely pays for decode.

**Context parallelism** splits a long sequence's tokens across GPUs. Each GPU holds the KV of
its chunk. Queries either travel around a ring meeting every chunk (ring attention), or each
GPU computes partial attention over its chunk and the partials are combined with the **LSE
merge from step [10](10-flashattention.md)**, which is exact.

**DeepSeek-style serving** combines them: MLA attention with data parallelism (the latent
cache is per-request, not per-head, step 5) and expert parallelism for MoE layers, with
specialized all-to-all kernels (DeepEP) and prefill/decode on separate nodes.

## 12.4 Prefill/decode disaggregation

**Problem.** In one engine, prefill and decode interfere. Chunked prefill bounds ITL but every
decode iteration still carries prompt chunks, and the two phases want different batch sizes,
parallelism and even GPU types.

**Core idea.** Run them on separate pools. Prefill workers compute the prompt's KV and send it
to a decode worker, which streams tokens. TTFT is set by the prefill pool, ITL by the decode
pool, and each is sized independently (DistServe measures goodput, meaning requests meeting
both SLOs).

The cost is one KV transfer per request:

$$
t_{\text{xfer}} = \frac{T_p \cdot m_{\text{tok}}}{\beta_{\text{link}}},\qquad
\text{worth it when } \frac{t_{\text{xfer}}}{t_{\text{prefill}}} \ll 1 \text{ (and it can overlap layer by layer)}.
$$

<span class="tag theory">theoretical</span> from `inference_lab.disagg`, 8K-token prompt,
prefill timed by the H100 roofline (Llama-3-70B: 1.26 s at peak):

| Link (nominal, per GPU) | 70B: transfer time | 70B: transfer / prefill | 8B: transfer / prefill |
|---|---|---|---|
| NVLink 4 (450 GB/s) | 6.0 ms | 0.5% | 1.6% |
| PCIe Gen5 x16 (64 GB/s) | 42 ms | 3.3% | 11% |
| InfiniBand NDR 400G (50 GB/s) | 54 ms | 4.3% | 14% |
| RoCE 200G (25 GB/s) | 107 ms | 8.5% | 29% |
| 100GbE TCP (12.5 GB/s) | 215 ms | 17% | 57% |

Prefill FLOPs grow with $P \cdot T_p$ while KV bytes grow with $m_{\text{tok}} \cdot T_p$, so
the ratio depends on $m_{\text{tok}}/P$. Big models with GQA or MLA have small ratios.
Disaggregation pays off on RDMA fabrics for large models, and is hard to justify for small
models on commodity Ethernet.

**State of the art.** Mooncake (Moonshot AI, serving Kimi) builds serving around a distributed
KV store spanning GPU, DRAM and SSD. NVIDIA Dynamo provides disaggregated serving with KV-aware
routing and NIXL for transfers. vLLM and SGLang support P/D disaggregation with pluggable KV
connectors (LMCache, Mooncake, NIXL).

## 12.5 Production serving engines

| Engine | Distinguishing design | Strong at |
|---|---|---|
| vLLM | PagedAttention; V1 scheduler with chunked prefill, prefix caching on by default; wide hardware and model support | general-purpose serving; ecosystem |
| SGLang | RadixAttention; fast CPU scheduler with overlap; DP-attention + EP for DeepSeek | prefix-heavy and agentic workloads; MoE at scale |
| TensorRT-LLM | compiled engines, in-flight batching, FP8/FP4 kernels for NVIDIA GPUs | peak performance on NVIDIA hardware |
| NVIDIA Dynamo | orchestration layer over engines: disaggregation, KV routing, KV offload | multi-node, datacenter scale |

Cross-cutting techniques in all of them: **CUDA graphs** (capture the decode step once, replay
it without per-kernel launch overhead, which matters because decode kernels are microseconds
long), fused kernels (RMSNorm + quantize, fused MoE), and overlapping CPU scheduling with GPU
execution.

## Questions

1. Why does INT4 weight-only quantization speed up decode much more than prefill?
2. Prove that the speculative-sampling output distribution equals $p$ for one position.
3. Why is TP all-reduce latency-bound in decode but bandwidth-bound in prefill?
4. Disaggregation for an 8B model over 100GbE: worth it? Use the table and say what would
   change your answer.

<div class="ie-prov" markdown>

**Provenance**

Source
: GPTQ (Frantar et al., 2022), AWQ (Lin et al., MLSys 2024), SmoothQuant (Xiao et al., ICML
  2023), FP8 formats (Micikevicius et al., 2022), OCP microscaling (MX) spec. Speculative
  decoding: Leviathan et al. (ICML 2023), Chen et al. (2023); Medusa (Cai et al., 2024); EAGLE
  (github.com/SafeAILab/EAGLE, linked only); DFlash (arXiv:2602.06036); DSpark (arXiv:2607.05147);
  DeepSpec (github.com/deepseek-ai/DeepSpec, MIT). Megatron-LM tensor parallelism (Shoeybi et al.,
  2019). Ring Attention (Liu et al., 2023). DistServe (Zhong et al., OSDI 2024). Mooncake
  (github.com/kvcache-ai/Mooncake). NVIDIA Dynamo (github.com/ai-dynamo/dynamo, linked only).
  DeepEP (github.com/deepseek-ai/DeepEP).

Derivation
: The speculative-sampling proof follows Leviathan et al. The disaggregation ratio and the
  engine comparison are this site's summaries.

Implementation
: `inference_lab.specdec` (PyTorch: acceptance rule with a distribution test, payoff formula,
  confidence-scheduled verification), `inference_lab.disagg` (transfer model).

Experiment
: Monte Carlo verification of the speculative formulas (CPU). No multi-GPU measurements
  were run.

</div>
