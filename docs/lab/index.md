---
title: The lab
description: >-
  Interactive calculators, simulators and runnable notebooks for LLM inference engineering, backed
  by a tested Python package.
---

# The lab

<p class="lede">Everything on this site that produces a number lives here, in three forms. The
same formulas run in your browser as interactive tools, in a small tested Python package, and
in notebooks that run each experiment end to end.</p>

| | What it is | Where |
|---|---|---|
| [Interactive tools](tools.md) | KV cache calculator, paged allocator, prefix-sharing simulator, batching timeline, speculative decoding payoff | in the browser, no install |
| `inference_lab` | 11 small modules: KV math, paged allocator, prefix caches, scheduler, IO-aware attention, MLA absorption, KV quantization, eviction, speculative decoding, disaggregation | [`src/inference_lab`](https://github.com/supriyo100/kv-cache-attention-variants/tree/main/src/inference_lab), 37 tests in [`tests/`](https://github.com/supriyo100/kv-cache-attention-variants/tree/main/tests) |
| [Notebooks](notebooks.md) | original notebooks, separate from the upstream course's `teaching_notebooks/` | [`notebooks/`](https://github.com/supriyo100/kv-cache-attention-variants/tree/main/notebooks) |

## Run it locally

```bash
git clone https://github.com/supriyo100/kv-cache-attention-variants
cd kv-cache-attention-variants
uv sync --extra dev
uv run pytest                                  # 37 tests, CPU only, seconds
uv run jupyter lab notebooks/                  # open any notebook
uv run python tools/build_notebooks.py         # regenerate notebooks (add --execute to run them)
```

Nothing here needs a GPU. The notebooks are published **without outputs**: run them on your
own machine or on Colab. Tables on the site come from the same small, deterministic computations
the unit tests run. Experiments that only make sense on a GPU (kernel timings, real TTFT) are
marked as such and are not run.

## Measured, theoretical, expected

| Label | Meaning | Example |
|---|---|---|
| <span class="tag measured">measured</span> | the code ran; the hardware is named | fragmentation simulation (CPU), KV quantization error (CPU) |
| <span class="tag theory">theoretical</span> | a formula with datasheet inputs | roofline step times, scheduler latencies |
| <span class="tag trend">expected trend</span> | the direction published work reports; no number of ours | quality impact of 2-bit KV on real models |
