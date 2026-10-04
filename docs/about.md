---
title: About
description: >-
  About this inference engineering notebook and its author, Supriyo Chakraborty: what is original
  here, how results are labelled, and how to get in touch.
---

# About

<p class="lede">This site is a public engineering notebook on LLM inference. I wrote it to
understand the field properly: to derive the math myself, rebuild each mechanism in small
tested code, and record the trade-offs in a form other engineers can check.</p>

## Author

**Supriyo Chakraborty**, India. Affiliation listed on GitHub: IIT Kharagpur.

Other public work on [GitHub](https://github.com/supriyo100) includes deep learning for
seismic inversion (frequency-domain U-Nets), mixed-integer optimization for manufacturing
scheduling and inventory planning, and agentic AI pipelines for automated research reports.

<!-- LinkedIn: set extra.author.linkedin in mkdocs.yml; the footer and page metadata pick it up. -->

- GitHub: [github.com/supriyo100](https://github.com/supriyo100)
- This site's source: [supriyo100/kv-cache-attention-variants](https://github.com/supriyo100/kv-cache-attention-variants)

## What is original here, and what is not

| Part | Status |
|---|---|
| The 12-step path, comparisons, SOTA map and lab pages | written for this site |
| Inline figures and interactive tools | original |
| `src/inference_lab` (11 modules, 37 tests) and `notebooks/` | original code |
| Excalidraw drawings, `beyond/` and `study/` notes | original (the earlier edition of this fork) |
| `docs/sessions/`, `src/kv_cache_variants/`, `teaching_notebooks/`, diagrams D01–D12 | from the upstream course; see [Sources and attribution](resources.md) |
| The techniques themselves (PagedAttention, FlashAttention, MLA, ...) | the cited authors' work; this site explains and rebuilds them, it does not claim them |

## How to read results

<span class="tag measured">measured</span> means the code ran, on named hardware (here: CPU
simulations and numerical checks). <span class="tag theory">theoretical</span> means a formula
with datasheet inputs. <span class="tag trend">expected trend</span> means a direction reported
in published work, with no number of ours. No GPU benchmark on this site is invented. Where a
result needs hardware I don't have, the code is provided and the number is left out.

## Corrections

Found an error in a derivation, a number or a claim?
[Open an issue](https://github.com/supriyo100/kv-cache-attention-variants/issues/new). Fixes
are welcome as pull requests.
