---
title: "Prefill vs decode: arithmetic intensity, the roofline and why decode waits on memory"
description: >-
  Why LLM prefill is compute-bound and decode is memory-bandwidth-bound: FLOP and byte counts
  per step, arithmetic intensity, the H100 roofline, the batch-size ceiling set by KV reads, and
  which optimizations help which phase.
step: 2
layer: hardware
---

# 2. Prefill and decode

<p class="lede">The same weights run two very different workloads. Prefill pushes thousands of
tokens through each weight matrix at once and keeps the tensor cores busy. Decode pushes one
token per sequence and spends most of its time moving weights and cached keys out of memory.
This step counts the FLOPs and bytes of each, puts both on a roofline, and shows why the two
phases need different optimizations.</p>

## Problem

On an H100, Llama-3-8B prefills an 8K-token prompt in about 150 ms, roughly 55,000 tokens/s.
The same GPU, decoding one sequence, produces about 200 tokens/s. That is a 270× gap on the
same hardware, the same model and the same matmuls. Why?

## The naive explanation, and why it fails

"Decode is sequential, so it can't be parallel." That is true across *steps*, but each decode
step is itself a big parallel computation, a pass through 8 billion parameters. The real
reason is not parallelism. It is the ratio of arithmetic to memory traffic.

## Core idea: arithmetic intensity

A kernel's arithmetic intensity is FLOPs performed per byte moved from HBM:
$I = F / Q$. A GPU with peak compute $\pi$ and bandwidth $\beta$ is compute-bound when
$I > \pi/\beta$ and bandwidth-bound below it. The threshold $I^{*} = \pi/\beta$ is the *ridge
point*.

| GPU (datasheet, dense) | $\pi$ BF16 | $\beta$ HBM | Ridge $I^{*}$ |
|---|---|---|---|
| A100 80GB SXM | 312 TFLOP/s | 2.04 TB/s | 153 FLOP/byte |
| H100 SXM | 989 TFLOP/s | 3.35 TB/s | 295 FLOP/byte |
| H200 SXM | 989 TFLOP/s | 4.8 TB/s | 206 FLOP/byte |
| B200 | 2250 TFLOP/s | 8.0 TB/s | 281 FLOP/byte |

The ridge has stayed between 150 and 300 FLOP/byte for three generations: compute grew about
as fast as bandwidth. So a kernel needs roughly 200+ FLOPs per byte to use the tensor cores.

## The math

A forward pass multiplies activations by every weight matrix. A token through a matrix with
$n$ parameters costs $2n$ FLOPs (one multiply and one add per parameter), so a pass over $t$
tokens costs about $2Pt$ FLOPs in the linear layers. The weights, $sP$ bytes, are read from HBM
once per pass whatever $t$ is.

**Prefill** of $T$ prompt tokens (one sequence):

$$
F_{\text{prefill}} \approx 2PT + \underbrace{2 L H_q d_h T^2}_{\text{causal attention}},
\qquad Q_{\text{prefill}} \approx sP + T\,m_{\text{tok}}
$$

$$
I_{\text{prefill}} \approx \frac{2PT}{sP} = \frac{2T}{s} \;\xrightarrow{\;s=2\;}\; T \text{ FLOP/byte}.
$$

A prompt longer than about 300 tokens is compute-bound on an H100.

**Decode**, one new token for each of $B$ sequences with context $T$:

$$
F_{\text{decode}} \approx 2PB + 4 L H_q d_h B T,
\qquad Q_{\text{decode}} \approx sP + B\,T\,m_{\text{tok}}
$$

$$
I_{\text{decode}} \approx \frac{2PB}{sP + B T m_{\text{tok}}}.
$$

Two limits matter:

- **Small batch** ($BTm_{\text{tok}} \ll sP$): $I \approx 2B/s = B$. One sequence gives
  $I \approx 1$, so the GPU runs at about $1/295$ of its compute peak and the step time is
  $t \approx sP/\beta$, the time to stream the weights once.
- **Large batch** ($B \to \infty$): $I \to \dfrac{2P}{T\,m_{\text{tok}}}$. Batching can no
  longer help, because every sequence brings its own KV to read. For Llama-3-8B at $T = 2048$:
  $2 \cdot 8\times10^9 / (2048 \cdot 131072) \approx 60$ FLOP/byte, still far below the ridge.

That second limit is the key result of this step. **Batching amortizes weight reads but not KV
reads.** Past a point, decode throughput is set by how many bytes of KV each token must read,
which is exactly what GQA, MLA and KV quantization (steps [5](05-attention-variants.md),
[11](11-long-context.md)) attack.

## The roofline

<figure class="ie-fig">
<svg viewBox="0 0 760 270" role="img" aria-label="Log-log roofline for H100: decode points sit on the bandwidth slope far left of the ridge at 295 FLOP per byte; prefill points sit on the flat compute roof">
<line x1="70" y1="230" x2="725" y2="230" class="ln"/><line x1="70" y1="230" x2="70" y2="15" class="ln"/>
<g class="t-sm">
<line x1="112.4" y1="230" x2="112.4" y2="235" class="ln"/><text x="112.4" y="248" text-anchor="middle" class="t-sm">1</text>
<line x1="253.7" y1="230" x2="253.7" y2="235" class="ln"/><text x="253.7" y="248" text-anchor="middle" class="t-sm">10</text>
<line x1="395" y1="230" x2="395" y2="235" class="ln"/><text x="395" y="248" text-anchor="middle" class="t-sm">100</text>
<line x1="536.3" y1="230" x2="536.3" y2="235" class="ln"/><text x="536.3" y="248" text-anchor="middle" class="t-sm">1,000</text>
<line x1="677.6" y1="230" x2="677.6" y2="235" class="ln"/><text x="677.6" y="248" text-anchor="middle" class="t-sm">10,000</text>
<text x="398" y="266" text-anchor="middle" class="t-sm">arithmetic intensity (FLOP / byte), log scale</text>
<text x="62" y="216" text-anchor="end" class="t-sm">1</text><text x="62" y="158" text-anchor="end" class="t-sm">10</text>
<text x="62" y="100" text-anchor="end" class="t-sm">100</text><text x="62" y="41" text-anchor="end" class="t-sm">1,000</text>
<text x="16" y="125" transform="rotate(-90 16 125)" text-anchor="middle" class="t-sm">TFLOP/s attained</text>
</g>
<path d="M70,199.4 L461.4,37.8 L720,37.8" class="ln" style="stroke:var(--ie-ink);stroke-width:2"/>
<line x1="461.4" y1="37.8" x2="461.4" y2="230" class="ln-d"/>
<text x="466" y="222" class="t-sm">ridge 295</text>
<text x="590" y="30" class="t-sm">989 TFLOP/s (BF16 peak)</text>
<text x="190" y="128" class="t-sm" transform="rotate(-22.4 190 128)">3.35 TB/s × intensity</text>
<line x1="363.6" y1="60" x2="363.6" y2="230" class="ln-d" style="stroke:var(--ie-waste)"/>
<text x="358" y="56" text-anchor="end" class="t-sm" style="fill:var(--ie-waste)">KV ceiling at T = 2048: ≈ 60</text>
<circle cx="112.4" cy="181.9" r="5" class="f-memory"/><text x="120" y="196" class="t-sm">decode B=1</text>
<circle cx="302.6" cy="103.3" r="5" class="f-memory"/><text x="262" y="96" class="t-sm">B=32</text>
<circle cx="354.4" cy="82" r="5" class="f-memory"/><text x="318" y="76" class="t-sm">B=256</text>
<circle cx="495.3" cy="37.8" r="5" class="f-compute"/><text x="490" y="60" class="t-sm">prefill 512</text>
<circle cx="667.2" cy="37.8" r="5" class="f-compute"/><text x="640" y="60" class="t-sm">prefill 8192</text>
</svg>
<figcaption><b>Llama-3-8B, BF16, H100 SXM, roofline model</b>
<span class="tag theory">theoretical</span>. Each point is the arithmetic intensity of one forward
step from <code>kv_math.step_cost</code>, placed on the H100 roof. Decode climbs the bandwidth
slope as batch grows, but stops at the dashed KV ceiling, about 60 FLOP/byte at a context of
2,048. It never reaches the ridge. Prefill sits on the compute roof once the prompt passes a
few hundred tokens.</figcaption>
</figure>

## Experiment

<span class="tag theory">theoretical</span> Roofline step times for Llama-3-8B, BF16, H100 SXM,
from `inference_lab.kv_math` (`step_cost` + `roofline_time`). Real kernels reach roughly 60–80%
of these bounds, so the trend matters more than the digits.

| Workload | Intensity (FLOP/B) | Step time | Bound | Tokens/s |
|---|---|---|---|---|
| Decode, $B=1$, $T=2048$ | 1.0 | 4.86 ms | memory | 206 |
| Decode, $B=8$ | 7.5 | 5.42 ms | memory | 1,477 |
| Decode, $B=32$ | 22.2 | 7.34 ms | memory | 4,359 |
| Decode, $B=128$ | 43.4 | 15.0 ms | memory | 8,512 |
| Decode, $B=256$ | 51.6 | 25.3 ms | memory | 10,119 |
| Prefill, $T=128$ | 128 | 4.78 ms | memory | 26,800 |
| Prefill, $T=512$ | 513 | 8.35 ms | compute | 61,300 |
| Prefill, $T=8192$ | 8,442 | 150 ms | compute | 54,500 |

Read across the decode rows. Going from $B=1$ to $B=32$ multiplies throughput by 21 while
step time grows only 1.5×, because the weights are read once for everyone. From $B=128$ to
$B=256$, throughput gains just 19% while step time grows 68%: KV reads now dominate the bytes.
That knee is where the per-user ITL target, not the GPU, decides the batch size.

## Different phases, different optimizations

| Technique | Helps prefill? | Helps decode? | Why |
|---|---|---|---|
| FP8 tensor cores | yes, about 2× peak FLOPs | a little | prefill is compute-bound |
| INT4/FP8 weight-only quantization | little | yes | decode streams the weights each step |
| GQA / MLA / KV quantization | little | yes, at large $B$ or $T$ | shrinks $m_{\text{tok}}$, the KV ceiling |
| Larger batch | no (already saturated) | yes, until the KV ceiling | amortizes weight reads |
| Prefix caching | yes, skips work | no | removes prefill FLOPs |
| Speculative decoding | no | yes, at small $B$ | turns idle FLOPs into extra tokens per weight read |
| Tensor parallelism | yes | yes | splits both FLOPs and bytes, adds communication |
| Chunked prefill / disaggregation | scheduling | protects ITL | stops the two phases interfering (steps [9](09-scheduling.md), [12](12-scaling-out.md)) |

## Failure modes

- **Optimizing the wrong phase.** INT4 weights make a summarization workload (long prompts,
  short outputs) barely faster, because it is prefill-dominated.
- **Mistaking low utilization for inefficiency.** A decode kernel at 3% of peak FLOPs can be
  at 85% of peak bandwidth, which is optimal. Measure achieved bandwidth, not FLOPs.
- **MoE decode.** For one token only the active experts' weights are read. At large batch,
  different tokens route to different experts, so most experts get touched and bytes per step
  approach the full parameter count.

## Questions

1. Derive the batch size at which decode KV reads equal weight reads for Llama-3-70B at
   $T = 8192$.
2. Why does the decode KV ceiling fall as context grows, and what does that imply for
   long-context serving?
3. An H200 has the same FLOPs as an H100 but 43% more bandwidth. Which phase gets faster?
4. Why does speculative decoding help at batch 1 but not at batch 256?

<div class="ie-prov" markdown>

**Provenance**

Source
: Roofline model (Williams, Waterman, Patterson, CACM 2009). GPU numbers are NVIDIA datasheets.
  The $2P$ FLOPs-per-token estimate is standard (Kaplan et al., 2020).

Derivation
: The decode intensity limit $2P/(T m_{\text{tok}})$ and the table are derived here.

Implementation
: `inference_lab.kv_math.step_cost`, `roofline_time` (tested:
  `test_decode_is_memory_bound_and_prefill_compute_bound`).

Experiment
: Theoretical only. Notebook `notebooks/12_prefill_vs_decode.ipynb` regenerates the table and
  plot. No GPU timing was run.

</div>
