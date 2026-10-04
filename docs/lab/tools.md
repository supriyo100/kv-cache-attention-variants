---
title: "Interactive tools: KV cache calculator, paged allocator and batching simulators"
description: >-
  Browser-based tools for LLM inference: a KV cache memory calculator for any model and GPU, a
  paged KV block allocator, a prefix-sharing and reference-count simulator, a static vs
  continuous batching timeline, and a speculative decoding speedup calculator.
hide:
  - toc
---

# Interactive tools

<p class="lede">Five tools that run in your browser. Each one uses the same formulas as the tested
Python in <code>inference_lab</code> and links to the step that derives them.</p>

## KV cache calculator

Bytes per token, per sequence and per batch; share of GPU memory; maximum concurrent sequences;
and the bandwidth bound on decode speed. Derivation: [step 4](../path/04-kv-cache.md).

<div data-widget="kvcalc"></div>

## Paged KV allocator

A 64-block pool. Admit, decode, finish, and watch block tables, internal fragmentation and
preemption. Derivation: [step 7](../path/07-paged-kv.md).

<div data-widget="allocator"></div>

## Prefix sharing and reference counts

Shared system prompt, private questions, block counts with and without sharing. Derivation:
[step 8](../path/08-prefix-sharing.md).

<div data-widget="prefix"></div>

## Static vs continuous batching

Slot occupancy over iterations for the same sixteen requests under both policies. Derivation:
[step 9](../path/09-scheduling.md).

<div data-widget="batching"></div>

## Speculative decoding payoff

Expected tokens per target pass and wall-time speedup as functions of acceptance rate, draft
length and draft cost. Derivation: [step 12](../path/12-scaling-out.md).

<div data-widget="specdec"></div>
