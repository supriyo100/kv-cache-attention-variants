# Inference Engineering: from tokens to GPU systems

<p>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/framework-PyTorch-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/docs-Material%20for%20MkDocs-526cfe?logo=markdown&logoColor=white" alt="MkDocs">
  <img src="https://github.com/supriyo100/kv-cache-attention-variants/actions/workflows/docs.yml/badge.svg" alt="Test, build and deploy">
</p>

A public engineering notebook on LLM inference. It follows one request from arrival to its last
token and stops wherever something becomes the bottleneck: prefill vs decode, the KV cache,
MHA/GQA/MLA, fragmentation, paged memory, prefix sharing, scheduling, FlashAttention, long
context, quantization, speculative decoding and distributed serving. Each step has the
derivation, a diagram, tested PyTorch code, the trade-offs and the current state of the art.

**Site:** https://supriyo100.github.io/kv-cache-attention-variants/

## What is here

| Part | What | Where |
|---|---|---|
| The visual story | 17 generated plates in six acts, one request from prompt to the state of the art, each linked to its hand-drawn sheet | [`docs/story.md`](docs/story.md) |
| The path | 12 steps, one request end to end, every topic as problem → math → design → code → trade-offs → SOTA | [`docs/path/`](docs/path/) |
| State of the art | the 2026 landscape by layer, and each technique's most-starred open-source implementation | [`docs/sota/`](docs/sota/) |
| Comparisons | 11 head-to-head matrices (MHA vs GQA vs MLA, paged vs contiguous, FP16 vs FP8 vs INT4, ...) | [`docs/compare/`](docs/compare/) |
| Interactive tools | KV calculator, paged allocator, prefix-sharing, batching and speculative decoding simulators (vanilla JS) | [`docs/assets/javascripts/widgets.js`](docs/assets/javascripts/widgets.js) |
| `inference_lab` | 11 small PyTorch/Python modules: KV math, paged allocator with refcounts and copy-on-write, prefix caches (hash-chained and radix), scheduler, tiled attention and LSE merge, MLA absorption on the course's MLA module, KIVI and TurboQuant, KV eviction, speculative decoding, disaggregation | [`src/inference_lab/`](src/inference_lab/), 37 tests in [`tests/`](tests/) |
| Notebooks | 14 original notebooks (6 core, 8 SOTA), published without outputs | [`notebooks/`](notebooks/) |
| References | 44 arXiv citations verified against the arXiv API | [`tools/references.py`](tools/references.py) → [`docs/references.md`](docs/references.md) |
| Explanation → implementation | for every idea: where it is explained here, the reference module, and the usable component in PyTorch, transformers, vLLM, SGLang, FlashAttention, FlashInfer, kvpress | [`docs/implementation.md`](docs/implementation.md) |
| Plan | audit, gap analysis, architecture, milestones, risks | [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) |

### Original vs upstream

This repository started as a fork of
[sourangshupal/kv-cache-attention-variants](https://github.com/sourangshupal/kv-cache-attention-variants).
The upstream course material is kept and credited: `docs/sessions/`, `src/kv_cache_variants/`,
`teaching_notebooks/` and diagrams D01–D12. Everything listed in the table above, plus the
Excalidraw drawings and the `docs/beyond/` and `docs/study/` notes, is original. Reference
implementations (vLLM, SGLang, nano-vllm, FlashAttention, FlashMLA, LMCache, StreamingLLM,
DeepSpec, the TurboQuant port) are studied and cited; no code is copied from them. See
[Sources and attribution](docs/resources.md).

## Run it

The project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev                       # project + pytest, ruff, arxiv
uv run pytest                             # 37 tests, CPU only, seconds
uv run jupyter lab notebooks/             # notebooks (run them on your machine or Colab)

uv run python tools/build_notebooks.py    # regenerate notebooks (add --execute to also run them)
uv run python tools/references.py         # re-verify citations and regenerate docs/references.md
uv run python tools/social_card.py        # regenerate the OpenGraph image

uv sync --extra docs && uv run mkdocs serve   # site at http://127.0.0.1:8000
```

Upstream course walkthroughs still work:
`uv run python -m kv_cache_variants.lessons.session05_gqa`.

After editing an Excalidraw drawing in `docs/assets/excalidraw/`, regenerate its SVG:
`cd tools/excalidraw_export && npm install && node export.mjs` (needs Node and Chrome or Edge).

The story plates (`docs/assets/plates/*.svg`, used by [The visual story](docs/story.md) and
[Attention variants compared](docs/beyond/variants_compared.md)) are generated, with every number
computed: `uv run python tools/story_plates.py`. `hooks/embeds.py` inlines them and turns each
Excalidraw sheet into a pan/zoom poster with section jumps (`page.html#sheet-<name>@<n>` deep-links a section).

## How results are labelled

- **measured**: the code ran; the hardware is named (here: CPU simulations and numerical checks).
- **theoretical**: a formula with datasheet inputs (roofline step times, scheduler latencies).
- **expected trend**: the direction published work reports, with no number of ours.

No GPU benchmark here is invented. Where a result needs hardware this project hasn't used, the code is
provided and the number is left out.

## Deploy

GitHub Actions (`.github/workflows/docs.yml`) runs lint and tests, builds the site with
`mkdocs build --strict`, and deploys to GitHub Pages on every push to `main`. One-time setup:
**Settings → Pages → Source: GitHub Actions**.

## Contributing

Corrections to derivations, numbers or claims are very welcome:
[open an issue](https://github.com/supriyo100/kv-cache-attention-variants/issues/new) or a pull request.
