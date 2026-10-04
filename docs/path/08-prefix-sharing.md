---
title: "Prefix caching and block sharing: reference counts, copy-on-write and radix trees"
description: >-
  How LLM servers share KV cache across requests: hash-chained block caching (vLLM), radix trees
  (SGLang RadixAttention), reference counting, block ownership and lifecycle, copy-on-write, LRU
  eviction, cache-aware routing, and why duplication is a different problem from fragmentation.
step: 8
layer: cache sharing
---

# 8. Prefix sharing

<p class="lede">Fifty users of the same assistant send the same 2,000-token system prompt.
Without sharing, the server computes and stores those 2,000 tokens of KV fifty times. Paging
makes it possible to store them once: two block tables can point at the same physical block.
The hard parts are knowing when two prefixes are the same, and knowing when a shared block can
be freed.</p>

## Problem

Three kinds of redundancy are common in real traffic:

1. **Shared system prompts and few-shot examples.** Every request of an application starts with
   the same tokens.
2. **Multi-turn chat.** Turn $n$'s prompt is turn $n-1$'s prompt plus the reply plus a new
   message.
3. **Parallel sampling, beam search and agents.** Many continuations branch from one context.

Each redundant prefix costs memory (duplicate KV) and compute (duplicate prefill, which is
TTFT).

## The naive solution, and why it fails

"Detect identical prompts and reuse the whole KV." Exact repeats are rare. What repeats is a
*prefix*, and it is followed by different suffixes. KV for token $j$ depends on tokens $1..j$
(step [4](04-kv-cache.md)), so a cached block can be reused only by a request whose **entire
prefix up to that block** is identical. Matching on a block's own tokens alone would be wrong.

## Core idea 1: hash-chained blocks (vLLM automatic prefix caching)

Identify each *full* block by a hash of its parent's hash and its own tokens:

$$
h_0 = H(\varnothing,\ x_{1..B}),\qquad h_i = H\big(h_{i-1},\ x_{iB+1..(i+1)B}\big).
$$

Two blocks have the same $h_i$ only if (barring collisions) the whole prefix matches. A global
table maps $h \mapsto$ physical block. On admission the scheduler walks the prompt's blocks in
order and stops at the first miss; everything before it is a hit and skips prefill. Only full
blocks are shareable. The partial last block is private, because the next token written into
it would differ between requests.

## Core idea 2: a radix tree over tokens (SGLang RadixAttention)

Store cached sequences in a radix tree whose edges are labelled with runs of tokens. A new
request walks from the root and matches the longest path. When it diverges in the middle of an
edge, the edge is split. Matching is at token granularity, and the tree makes the sharing
structure explicit: system prompt at the root, conversations as branches, turns as deeper
nodes.

<figure class="ie-fig">
<svg viewBox="0 0 760 260" role="img" aria-label="Radix tree: root edge is the shared system prompt with lock count 3; it branches to users A, B and C; user A's branch has a second turn">
<rect x="20" y="105" width="150" height="44" class="shared"/><text x="95" y="124" text-anchor="middle" class="t-b">system prompt</text><text x="95" y="141" text-anchor="middle" class="t-sm">2,000 tok · lock 3</text>
<path d="M170,127 L250,45" class="ln"/><path d="M170,127 L250,127" class="ln"/><path d="M170,127 L250,210" class="ln"/>
<rect x="250" y="25" width="140" height="40" class="memory"/><text x="320" y="44" text-anchor="middle" class="t-b">A: question 1</text><text x="320" y="59" text-anchor="middle" class="t-sm">lock 1</text>
<rect x="250" y="107" width="140" height="40" class="memory"/><text x="320" y="126" text-anchor="middle" class="t-b">B: question</text><text x="320" y="141" text-anchor="middle" class="t-sm">lock 1</text>
<rect x="250" y="190" width="140" height="40" class="free"/><text x="320" y="209" text-anchor="middle" class="t-b">C: question</text><text x="320" y="224" text-anchor="middle" class="t-sm">lock 0 · LRU-evictable</text>
<path d="M390,45 L460,45" class="ln"/>
<rect x="460" y="25" width="170" height="40" class="memory"/><text x="545" y="44" text-anchor="middle" class="t-b">A: reply 1 + question 2</text><text x="545" y="59" text-anchor="middle" class="t-sm">lock 1 (turn 2 running)</text>
<text x="420" y="120" class="t-sm">A new request with the same system prompt</text>
<text x="420" y="136" class="t-sm">matches the root edge (2,000 tokens skip prefill),</text>
<text x="420" y="152" class="t-sm">then adds its own leaf.</text>
<text x="420" y="190" class="t-sm">C finished: its leaf stays cached (lock 0) and is</text>
<text x="420" y="206" class="t-sm">evicted first, leaf-first in LRU order, if memory</text>
<text x="420" y="222" class="t-sm">is needed. The root cannot go while anyone holds it.</text>
</svg>
<figcaption><b>Sharing structure as a tree.</b> <span class="k-shared">Shared</span> nodes carry a
lock (reference) count equal to the running requests using them. <span class="k-memory">Private</span>
suffixes are leaves. Finished requests leave their KV cached at lock 0, ready for a future hit,
until memory pressure evicts them.</figcaption>
</figure>

### How the reference implementations do it

The most-starred minimal engine, [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
(`nanovllm/engine/block_manager.py`, MIT), shows the production design in about 120 lines, and
`inference_lab.prefix.HashBlockCache` follows it:

- **The free list is the LRU.** Freed blocks go to the *tail* of a FIFO queue and keep their
  hash. A later request can still hit them until they reach the head and are reallocated, which
  is the moment of eviction. No separate LRU structure exists. Freeing a request's blocks in
  reverse order puts its prefix at the tail, so shared prefixes are evicted last.
- **A hit is verified.** The block's stored token IDs are compared with the request's, so a hash
  collision can never serve wrong KV.
- **The last block is never served from cache.** At least one token is always computed, because
  the forward pass must produce logits for the next token.

## Reference counting: who owns a shared block?

A shared block has no single owner. Each physical block carries a reference count:

$$
\text{ref}(b) = \big|\{(r, i) : \text{table}_r[i] = b\}\big|,
$$

the number of block-table entries, across all live requests, that point at $b$. The lifecycle:

| Event | Effect on ref counts |
|---|---|
| Admit with a cache hit on blocks $b_1..b_k$ | $\text{ref}(b_j) \mathrel{+}= 1$ (if it was 0, remove $b_j$ from the evictable list) |
| Allocate a new private block | new block, $\text{ref} = 1$ |
| Fork (parallel sampling) | every block of the parent $\mathrel{+}= 1$ |
| Write into a block with $\text{ref} > 1$ | **copy-on-write**: allocate a copy, $\text{ref}(\text{old}) \mathrel{-}= 1$, the writer points at the copy |
| Request finishes | every block in its table $\mathrel{-}= 1$ |
| $\text{ref}(b)$ reaches 0 | without caching: return to free list. With prefix caching: keep the contents, add to the LRU list, evict only when space is needed |

Without reference counts, freeing a finished request would free blocks that other running
requests still read. That is a use-after-free, and the symptom is not a crash but silently
corrupted attention. Copy-on-write matters for forks: parallel samples share a partial last
block until the first sample writes a new token, at which point it gets its own copy.

## Fragmentation vs duplication, again

Prefix sharing does not change fragmentation. It removes **duplication**. Every slot of a
duplicated prefix holds a valid token, just a token stored elsewhere already. The two fixes
compose: paging makes blocks interchangeable, and sharing makes identical blocks single.

## Try it

<div data-widget="prefix"></div>

## Experiment

<span class="tag measured">measured</span> (CPU simulation of the data structures; deterministic,
seed 0), using `inference_lab.prefix`:

| Workload | Metric | Hash-chained blocks ($B=16$) | Radix tree |
|---|---|---|---|
| 50 users, same 2,000-token system prompt + 200 private tokens | prompt tokens that skip prefill | 98,000 / 110,000 (89.1%) | 98,000 (89.1%) |
| same | KV storage | 775 blocks vs 6,900 without sharing | 12,000 tokens vs 110,000 |
| 20 conversations × 6 turns (500-token shared preamble, 150-token user turns, 250-token replies) | prompt tokens that skip prefill | 77.5% | 78.0% |

In the multi-turn case the radix tree wins slightly, because it matches inside a partial block,
while the hash cache reuses only full blocks. In this model replies are not inserted into the
cache, so turn $n+1$ re-prefills the previous reply. Engines that insert generated tokens on
completion, as SGLang does, hit higher.

## Production notes and state of the art

- **vLLM** enables automatic prefix caching by default in V1 (hash-chained blocks, LRU
  eviction of zero-refcount blocks). **SGLang** uses RadixAttention with leaf-first LRU
  eviction.
- **Cache-aware routing.** A prefix cache helps only if the request lands on the GPU holding
  the prefix. SGLang's router, NVIDIA Dynamo's KV-aware router and llm-d route by expected
  prefix overlap, balanced against load.
- **Hierarchical caches.** Evicted KV can move to CPU DRAM or SSD instead of being discarded,
  and come back on a hit: SGLang HiCache, LMCache, Mooncake's distributed KV store. This trades
  PCIe or network transfer time against recompute time (step [12](12-scaling-out.md)).
- **Provider prompt caching.** Commercial APIs expose prefix caching to users, with lower
  prices for cached input tokens. This is the same mechanism with an economic interface.
- **Isolation.** A shared cache leaks timing: a hit gives a faster TTFT. Published attacks use
  this to infer other tenants' prompts. Multi-tenant deployments isolate caches per tenant, for
  example with a per-request cache salt mixed into the block hash.

## Failure modes

- **Hash collisions.** A 64-bit hash collision would silently serve wrong KV. Production
  systems use long cryptographic hashes or verify tokens on hit.
- **Prompt formatting drift.** One different token early in a template (a timestamp, a
  request ID) breaks every later block's hash. Put variable content at the end.
- **Eviction thrash.** If the working set of hot prefixes exceeds cache memory, LRU evicts
  what is about to be reused, and hit rate collapses.

## Questions

1. Why must block hashes chain through the parent hash? Construct a wrong reuse that would
   happen without chaining.
2. Two parallel samples share a 6-token partial block of size 16. What happens, step by step,
   when each writes its next token?
3. Why does prefix caching reduce TTFT but not TPOT?
4. When is CPU offload of evicted prefixes better than recompute? Write the inequality.

<div class="ie-prov" markdown>

**Provenance**

Source
: Kwon et al., SOSP 2023 (sharing and copy-on-write in PagedAttention). Zheng et al.,
  "SGLang" (arXiv:2312.07104, RadixAttention; github.com/sgl-project/sglang, Apache-2.0).
  vLLM automatic prefix caching design docs. LMCache (github.com/LMCache/LMCache),
  Mooncake (github.com/kvcache-ai/Mooncake).

Derivation
: Reference-count definition and lifecycle table written for this site.

Implementation
: `inference_lab.prefix` (`HashBlockCache`, `RadixCache`) and `inference_lab.paged.fork`
  for copy-on-write. Tested: chained-hash position sensitivity, full-block-only sharing,
  cached-until-evicted, radix split and LRU eviction of unlocked leaves.

Experiment
: The table above, reproducible with `notebooks/sota_01_radix_prefix_cache.ipynb`.

</div>
