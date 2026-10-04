"""Paged KV cache: a block allocator with reference counting and copy-on-write.

Model of the PagedAttention memory manager (Kwon et al., SOSP 2023,
arXiv:2309.06180; github.com/vllm-project/vllm). A sequence's KV is split into
fixed-size logical blocks of ``block_size`` tokens. A per-sequence block table
maps logical block i to a physical block id. Physical blocks carry a reference
count so several sequences can point at the same block (shared prefix, parallel
sampling, beam search). Writing into a shared block first copies it
(copy-on-write).

Invariants (checked by tests/test_paged.py):
  * every physical block is either free (refcount 0, on the free list) or used;
  * refcount[b] == number of block-table entries that point at b;
  * a sequence's token count fits its table: (n_blocks-1)*B < tokens <= n_blocks*B.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class OutOfBlocks(RuntimeError):
    """No free physical block. The scheduler must queue, preempt, or swap."""


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        # Pop from the end; reversed so block 0 is handed out first (easier to read in demos).
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))
        self.refcount: list[int] = [0] * num_blocks
        self.copies = 0  # copy-on-write events

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocks("KV block pool exhausted")
        b = self._free.pop()
        self.refcount[b] = 1
        return b

    def incref(self, b: int) -> None:
        if self.refcount[b] <= 0:
            raise ValueError(f"block {b} is free; cannot share it")
        self.refcount[b] += 1

    def decref(self, b: int) -> None:
        if self.refcount[b] <= 0:
            raise ValueError(f"double free of block {b}")
        self.refcount[b] -= 1
        if self.refcount[b] == 0:
            self._free.append(b)


@dataclass
class Sequence:
    seq_id: str
    tokens: int = 0
    table: list[int] = field(default_factory=list)  # logical block -> physical block


class PagedKVCache:
    """Block tables for many sequences over one shared allocator."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.alloc = BlockAllocator(num_blocks, block_size)
        self.B = block_size
        self.seqs: dict[str, Sequence] = {}

    # -- lifecycle -------------------------------------------------------------------
    def add(self, seq_id: str, prompt_tokens: int) -> Sequence:
        """Admit a sequence and allocate blocks for its prompt (prefill)."""
        if seq_id in self.seqs:
            raise ValueError(f"duplicate sequence {seq_id}")
        need = -(-prompt_tokens // self.B)
        if need > self.alloc.num_free:
            raise OutOfBlocks(f"{seq_id} needs {need} blocks, {self.alloc.num_free} free")
        s = Sequence(seq_id)
        self.seqs[seq_id] = s
        s.table = [self.alloc.allocate() for _ in range(need)]
        s.tokens = prompt_tokens
        return s

    def append_token(self, seq_id: str) -> int | None:
        """Decode step: reserve a slot for one new token.

        Returns the newly allocated physical block if one was needed, else None.
        If the last block is shared, it is copied first (copy-on-write).
        """
        s = self.seqs[seq_id]
        if s.tokens % self.B == 0:  # last block full (or empty table)
            b = self.alloc.allocate()
            s.table.append(b)
            s.tokens += 1
            return b
        last = s.table[-1]
        if self.alloc.refcount[last] > 1:  # shared partial block: copy before writing
            new = self.alloc.allocate()
            self.alloc.decref(last)
            s.table[-1] = new
            self.alloc.copies += 1
        s.tokens += 1
        return None

    def fork(self, parent_id: str, child_id: str) -> Sequence:
        """Child shares every parent block (parallel sampling / beam search)."""
        p = self.seqs[parent_id]
        c = Sequence(child_id, p.tokens, list(p.table))
        for b in c.table:
            self.alloc.incref(b)
        self.seqs[child_id] = c
        return c

    def free(self, seq_id: str) -> None:
        s = self.seqs.pop(seq_id)
        for b in s.table:
            self.alloc.decref(b)

    # -- accounting ------------------------------------------------------------------
    def physical_slot(self, seq_id: str, pos: int) -> tuple[int, int]:
        """Translate a token position into (physical block, offset): the block-table lookup."""
        s = self.seqs[seq_id]
        if not 0 <= pos < s.tokens:
            raise IndexError(pos)
        return s.table[pos // self.B], pos % self.B

    def internal_waste_slots(self) -> int:
        """Reserved-but-unused token slots in last blocks (counted once per physical block)."""
        seen: set[int] = set()
        waste = 0
        for s in self.seqs.values():
            if s.table and s.table[-1] not in seen:
                seen.add(s.table[-1])
                waste += len(s.table) * self.B - s.tokens
        return waste

    def used_blocks(self) -> int:
        return self.alloc.num_blocks - self.alloc.num_free

    def check_invariants(self) -> None:
        counts = [0] * self.alloc.num_blocks
        for s in self.seqs.values():
            n = len(s.table)
            assert (n - 1) * self.B < s.tokens <= n * self.B or (n == 0 and s.tokens == 0)
            for b in s.table:
                counts[b] += 1
        assert counts == self.alloc.refcount, "refcount != table references"
        free = set(self.alloc._free)
        assert len(free) == len(self.alloc._free), "free list has duplicates"
        for b in range(self.alloc.num_blocks):
            assert (b in free) == (counts[b] == 0)
