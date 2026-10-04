---
title: Sources and attribution
description: >-
  Where the material on this site comes from: the upstream course this repository forks, the
  reference repositories, inspirations, official implementations and papers.
---

# Sources and attribution

<p class="lede">What came from where. Original work, upstream material and external references
are kept separate so readers can tell which is which.</p>

## Upstream course (forked)

This repository is a fork of
[sourangshupal/kv-cache-attention-variants](https://github.com/sourangshupal/kv-cache-attention-variants),
a course module from naive decoding to PagedAttention. From it come the session pages under
[Course (upstream)](sessions/index.md), the `src/kv_cache_variants` package, the
`teaching_notebooks/`, and the interactive diagrams D01–D12. That material belongs to its
author. The upstream repository does not currently state a license, and permission questions
should go to its author. This site's own code builds on the upstream modules (for example, the
MLA absorption code runs on the course's `MultiHeadLatentAttention`) and credits them where it
does.

## Inspiration

[elizabetht/100-days-of-inference](https://github.com/elizabetht/100-days-of-inference), a
daily series following Philip Kiely's *Inference Engineering* (Baseten Books, 2026), was used
as a checklist of topics worth covering. No text, structure, notebooks or figures were taken
from it.

## Reference implementations

Each technique is studied in its most-starred open-source implementation. The full table, with
star counts and the files where each idea lives, is on
[Techniques and implementations](sota/techniques.md). Repositories without a standard
open-source license (TensorRT-LLM, Dynamo, EAGLE, the GPL-3.0 TurboQuant port) are linked
only; no code from any reference repository is copied into this one.

## Papers

Every cited paper is listed on [References](references.md). Titles, authors and dates are
fetched from the arXiv API by `tools/references.py`, which fails if a cited ID does not match
its paper.

## Hardware figures

GPU peak FLOP/s, bandwidth and memory are NVIDIA datasheet values (dense, without sparsity).
They are upper bounds, used only in labelled theoretical estimates.
