"""Contiguous vs paged KV allocation: where the memory goes.

Three kinds of waste, measured in token slots:

* reservation waste: a contiguous allocator must reserve prompt + max_tokens
  slots up front because the final length is unknown; the slots past the
  current length sit idle while the request runs (and most are never used).
* internal fragmentation: paged allocation rounds each sequence up to a
  whole block, so waste per sequence is (-tokens) mod B, which is < B.
  If lengths are spread evenly modulo B, its mean is about (B-1)/2.
* external fragmentation: free memory exists in total, but no single hole is
  large enough for a contiguous request.

``simulate`` replays one arrival/finish trace under both allocators and counts
each kind of waste plus admission failures.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Request:
    rid: int
    arrive: int  # step at which it arrives
    prompt: int
    output: int  # true output length, unknown to the allocator
    max_new: int  # the request's declared max_tokens cap (output <= max_new)

    @property
    def reserve(self) -> int:
        """Slots a contiguous allocator must reserve: the worst case, prompt + max_new."""
        return self.prompt + self.max_new


def make_trace(n: int, seed: int = 0, prompt=(64, 1024), caps=(256, 512, 1024, 2048),
               gap: int = 3) -> list[Request]:
    """Random trace. Each request declares a max_tokens cap and stops somewhere below it."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        cap = rng.choice(caps)
        out.append(Request(i, i * gap, rng.randint(*prompt), rng.randint(16, cap), cap))
    return out


class ContiguousPool:
    """First-fit allocator over a 1-D slot array (like a malloc without compaction)."""

    def __init__(self, slots: int) -> None:
        self.slots = slots
        self.holes: list[tuple[int, int]] = [(0, slots)]  # (start, length), sorted

    def alloc(self, n: int) -> int | None:
        for i, (s, ln) in enumerate(self.holes):
            if ln >= n:
                self.holes[i] = (s + n, ln - n)
                if self.holes[i][1] == 0:
                    self.holes.pop(i)
                return s
        return None

    def free(self, start: int, n: int) -> None:
        self.holes.append((start, n))
        self.holes.sort()
        merged: list[tuple[int, int]] = []
        for s, ln in self.holes:
            if merged and merged[-1][0] + merged[-1][1] == s:
                merged[-1] = (merged[-1][0], merged[-1][1] + ln)
            else:
                merged.append((s, ln))
        self.holes = merged

    @property
    def free_total(self) -> int:
        return sum(ln for _, ln in self.holes)

    @property
    def largest_hole(self) -> int:
        return max((ln for _, ln in self.holes), default=0)


@dataclass
class Stats:
    admitted: int = 0
    rejected_external: int = 0  # enough total free memory, no contiguous hole
    rejected_full: int = 0  # not enough memory at all
    peak_live_tokens: int = 0
    mean_utilization: float = 0.0  # live tokens / allocated slots, averaged over steps
    mean_reservation_waste: float = 0.0
    mean_internal_waste: float = 0.0


def simulate(trace: list[Request], total_slots: int, mode: str, block_size: int = 16) -> Stats:
    """Replay ``trace``. mode: "contiguous" (reserve prompt + max_new) or "paged" (grow by blocks).

    Rejected requests are dropped (no retry) so both allocators see the same trace.
    One decode token per running request per step.
    """
    st = Stats()
    pool = ContiguousPool(total_slots)
    free_blocks = total_slots // block_size
    live: dict[int, dict] = {}
    pending = sorted(trace, key=lambda r: r.arrive)
    t = 0
    util_sum = res_sum = int_sum = 0.0
    steps = 0
    while pending or live:
        while pending and pending[0].arrive <= t:
            r = pending.pop(0)
            if mode == "contiguous":
                start = pool.alloc(r.reserve)
                if start is None:
                    if pool.free_total >= r.reserve:
                        st.rejected_external += 1
                    else:
                        st.rejected_full += 1
                    continue
                live[r.rid] = {"r": r, "tokens": r.prompt, "start": start, "blocks": 0}
            else:
                need = -(-r.prompt // block_size)
                if need > free_blocks:
                    st.rejected_full += 1
                    continue
                free_blocks -= need
                live[r.rid] = {"r": r, "tokens": r.prompt, "start": -1, "blocks": need}
            st.admitted += 1
        # decode one token for everyone; finish when output reached
        done = []
        for rid, s in live.items():
            r = s["r"]
            s["tokens"] += 1
            if mode == "paged" and s["tokens"] > s["blocks"] * block_size:
                if free_blocks == 0:  # simple policy: finish early (truncate) if pool is dry
                    done.append(rid)
                    continue
                free_blocks -= 1
                s["blocks"] += 1
            if s["tokens"] >= r.prompt + r.output:
                done.append(rid)
        live_tokens = sum(s["tokens"] for s in live.values())
        if mode == "contiguous":
            allocated = sum(s["r"].reserve for s in live.values())
            res_sum += allocated - live_tokens
        else:
            allocated = sum(s["blocks"] for s in live.values()) * block_size
            int_sum += allocated - live_tokens
        if allocated:
            util_sum += live_tokens / allocated
        st.peak_live_tokens = max(st.peak_live_tokens, live_tokens)
        steps += 1
        for rid in done:
            s = live.pop(rid)
            if mode == "contiguous":
                pool.free(s["start"], s["r"].reserve)
            else:
                free_blocks += s["blocks"]
        t += 1
    st.mean_utilization = util_sum / steps
    st.mean_reservation_waste = res_sum / steps
    st.mean_internal_waste = int_sum / steps
    return st


def internal_waste(tokens: int, block_size: int) -> int:
    """Unused slots in the last block: (-tokens) mod B. Always < B."""
    return (-tokens) % block_size
