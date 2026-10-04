"""Prefix caching: two ways to find KV that another request already computed.

1. Hash-chained blocks (the design of vLLM's automatic prefix caching).
   A *full* block is identified by hash(parent_hash, tokens_in_block), so a
   block's identity depends on its whole prefix, not just its own tokens.
   Lookup walks the prompt block by block and stops at the first miss.
   Only full blocks are shareable; the partial tail block is private.

2. Radix tree (the design of SGLang's RadixAttention, Zheng et al. 2024,
   arXiv:2312.07104; github.com/sgl-project/sglang). Edges are labelled
   with token runs of any length; a request matches the longest path from the
   root. Each node has a lock count (running requests using it); nodes with
   lock 0 stay cached and are evicted leaf-first in LRU order when memory is
   needed.

Both answer the same question: how many prompt tokens can skip prefill?
This is a model of the data structures, not of either codebase.
"""

from __future__ import annotations

import hashlib
import itertools
from collections import deque
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# 1. Hash-chained block cache
# --------------------------------------------------------------------------------------


def block_hashes(tokens: list[int], block_size: int) -> list[str]:
    """Chained hashes of the *full* blocks of ``tokens``."""
    out: list[str] = []
    parent = ""
    for i in range(len(tokens) // block_size):
        chunk = tokens[i * block_size:(i + 1) * block_size]
        h = hashlib.sha256((parent + ":" + ",".join(map(str, chunk))).encode()).hexdigest()[:16]
        out.append(h)
        parent = h
    return out


class HashBlockCache:
    """Block pool with hash-chained prefix reuse, structured like vLLM / nano-vllm.

    Design points taken from the reference implementations (vLLM ``vllm/v1/core/block_pool.py``;
    nano-vllm ``nanovllm/engine/block_manager.py``, the most-starred minimal engine):

    * the free list is a FIFO queue. A block whose refcount drops to 0 goes to the *tail*
      and keeps its hash, so it can still be hit until it reaches the head and is
      reallocated. The free list is therefore also the LRU eviction order; no separate
      LRU structure is needed.
    * on a hash hit the block's stored tokens are compared too, so a hash collision can
      never serve the wrong KV.
    * the last block of a prompt is never taken from the cache, so at least one token is
      always computed to produce the next-token logits.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.B = block_size
        self.capacity = num_blocks
        self.free: deque[int] = deque(range(num_blocks))
        self.ref = [0] * num_blocks
        self.block_hash: list[str | None] = [None] * num_blocks
        self.block_tokens: list[tuple[int, ...]] = [()] * num_blocks
        self.by_hash: dict[str, int] = {}
        self.hits = self.misses = self.evictions = 0

    def _allocate(self) -> int:
        if not self.free:
            raise RuntimeError("no free block")
        b = self.free.popleft()  # head = least recently freed
        old = self.block_hash[b]
        if old is not None and self.by_hash.get(old) == b:
            del self.by_hash[old]  # evict its cached contents
            self.evictions += 1
        self.block_hash[b] = None
        self.ref[b] = 1
        return b

    def acquire(self, tokens: list[int]) -> tuple[list[int], int]:
        """Block table for a prompt. Returns (table, number of tokens served from cache)."""
        n_blocks = -(-len(tokens) // self.B)
        hashes = block_hashes(tokens, self.B)
        table: list[int] = []
        cached = 0
        for i in range(n_blocks - 1):  # never reuse the last block
            h = hashes[i]
            b = self.by_hash.get(h)
            chunk = tuple(tokens[i * self.B:(i + 1) * self.B])
            if b is None or self.block_tokens[b] != chunk:
                break
            if self.ref[b] == 0:
                self.free.remove(b)  # revive a cached-but-free block
            self.ref[b] += 1
            table.append(b)
            cached += self.B
            self.hits += 1
        for i in range(len(table), n_blocks):
            b = self._allocate()
            self.misses += 1
            if i < len(hashes):  # full block: register it for future requests
                self.block_hash[b] = hashes[i]
                self.block_tokens[b] = tuple(tokens[i * self.B:(i + 1) * self.B])
                self.by_hash[hashes[i]] = b
            table.append(b)
        return table, cached

    def release(self, table: list[int]) -> None:
        for b in reversed(table):  # tail blocks first, so prefixes are evicted last
            self.ref[b] -= 1
            if self.ref[b] == 0:
                self.free.append(b)


# --------------------------------------------------------------------------------------
# 2. Radix tree (RadixAttention-style)
# --------------------------------------------------------------------------------------


@dataclass(eq=False)
class RadixNode:
    tokens: tuple[int, ...] = ()
    children: dict[int, RadixNode] = field(default_factory=dict)  # first token -> child
    parent: RadixNode | None = None
    lock: int = 0
    last_use: int = 0


class RadixCache:
    """Token-level radix tree. Capacity is counted in cached tokens."""

    def __init__(self, capacity_tokens: int) -> None:
        self.root = RadixNode()
        self.capacity = capacity_tokens
        self.size = 0
        self.clock = itertools.count(1)
        self.evicted_tokens = 0

    def match(self, tokens: list[int]) -> tuple[int, list[RadixNode]]:
        """Longest cached prefix. Returns (matched length, path of nodes)."""
        node, i, path = self.root, 0, []
        while i < len(tokens) and tokens[i] in node.children:
            child = node.children[tokens[i]]
            k = _common(child.tokens, tokens, i)
            if k < len(child.tokens):
                child = self._split(child, k)
            path.append(child)
            i += k
            node = child
        return i, path

    def insert(self, tokens: list[int]) -> tuple[int, list[RadixNode]]:
        """Match, then add the uncached suffix. Locks the whole path. Returns (hit tokens, path)."""
        hit, path = self.match(tokens)
        now = next(self.clock)
        node = path[-1] if path else self.root
        rest = tuple(tokens[hit:])
        if rest:
            self._make_room(len(rest), protect=set(map(id, path)))
            leaf = RadixNode(rest, parent=node)
            node.children[rest[0]] = leaf
            self.size += len(rest)
            path.append(leaf)
        for n in path:
            n.lock += 1
            n.last_use = now
        return hit, path

    def release(self, path: list[RadixNode]) -> None:
        for n in path:
            n.lock -= 1

    def _split(self, child: RadixNode, k: int) -> RadixNode:
        """Split child's edge after k tokens; returns the new upper node."""
        parent = child.parent
        assert parent is not None
        upper = RadixNode(child.tokens[:k], parent=parent, lock=child.lock, last_use=child.last_use)
        parent.children[upper.tokens[0]] = upper
        child.tokens = child.tokens[k:]
        child.parent = upper
        upper.children[child.tokens[0]] = child
        return upper

    def _leaves(self) -> list[RadixNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n is not self.root and not n.children:
                out.append(n)
            stack.extend(n.children.values())
        return out

    def _make_room(self, need: int, protect: set[int]) -> None:
        while self.size + need > self.capacity:
            cands = [n for n in self._leaves() if n.lock == 0 and id(n) not in protect]
            if not cands:
                raise RuntimeError("cache full of locked (in-use) prefixes")
            victim = min(cands, key=lambda n: n.last_use)
            assert victim.parent is not None
            del victim.parent.children[victim.tokens[0]]
            self.size -= len(victim.tokens)
            self.evicted_tokens += len(victim.tokens)

    def cached_tokens(self) -> int:
        total, stack = 0, [self.root]
        while stack:
            n = stack.pop()
            total += len(n.tokens)
            stack.extend(n.children.values())
        return total


def _common(edge: tuple[int, ...], tokens: list[int], start: int) -> int:
    k = 0
    while k < len(edge) and start + k < len(tokens) and edge[k] == tokens[start + k]:
        k += 1
    return k
