"""Iteration-level scheduling simulator: static vs continuous batching, chunked prefill.

Each iteration the scheduler picks which requests run, the model does one
forward pass, and every running request in decode emits one token. The pass is
timed with the roofline model in ``kv_math`` (max of compute time and HBM time),
so all latencies are *theoretical estimates* for the chosen model and GPU,
not measurements.

Policies
--------
static      Form a batch of up to ``max_batch`` waiting requests, prefill them
            together, decode until the *last* one finishes. Finished slots idle.
continuous  Orca-style iteration-level scheduling (Yu et al., OSDI 2022): admit
            new requests whenever blocks and batch slots allow. New prompts are
            prefilled in a prefill-only iteration that pauses running decodes
            (prefill-prioritising, as in early vLLM).
chunked     Continuous + chunked prefill (Sarathi-Serve, Agrawal et al., OSDI 2024;
            default in vLLM V1): each iteration has a token budget; decodes take
            one token each and the remainder is filled with prompt chunks, so
            decodes never stall behind a long prompt.

KV memory is paged (``block_size`` tokens per block, ``num_blocks`` total). A request's prompt
blocks are allocated at admission (admission control); decode grows one block at a time. When a
decode needs a block and none is free, the most recently admitted request is
preempted and later recomputed from scratch (vLLM's default recompute policy).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import kv_math


@dataclass
class Req:
    rid: int
    arrival: float
    prompt: int
    output: int
    # state
    prefilled: int = 0
    generated: int = 0
    blocks: int = 0
    first_token: float | None = None
    finish: float | None = None
    token_times: list[float] = field(default_factory=list)
    preemptions: int = 0

    @property
    def context(self) -> int:
        return self.prefilled + self.generated



def poisson_trace(n: int, rate: float, seed: int = 0, prompt=(128, 2048),
                  output=(32, 512)) -> list[Req]:
    rng = random.Random(seed)
    t, out = 0.0, []
    for i in range(n):
        t += rng.expovariate(rate)
        out.append(Req(i, t, rng.randint(*prompt), rng.randint(*output)))
    return out


@dataclass
class Result:
    policy: str
    makespan: float
    throughput_tok_s: float
    ttft_mean: float
    ttft_p99: float
    tpot_mean: float
    itl_p99: float
    preemptions: int
    mean_batch: float
    iterations: int


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def simulate(trace: list[Req], policy: str = "chunked", model: kv_math.ModelShape = kv_math.LLAMA3_8B,
             gpu: kv_math.GPU = kv_math.H100_SXM, num_blocks: int = 4096, block_size: int = 16,
             max_batch: int = 64, token_budget: int = 512) -> Result:
    reqs = [Req(r.rid, r.arrival, r.prompt, r.output) for r in trace]
    waiting = sorted(reqs, key=lambda r: r.arrival)
    running: list[Req] = []
    free = num_blocks
    now = 0.0
    iters = 0
    batch_sum = 0
    preempt = 0
    kvb = kv_math.kv_bytes_per_token(model)
    wbytes = kv_math.weight_bytes_active(model)

    def blocks_for(tokens: int) -> int:
        return -(-tokens // block_size)

    def step_time(new_tokens: int, ctx_read: int, attn_flops: float) -> float:
        flops = 2 * model.active * new_tokens + attn_flops
        bytes_ = wbytes + ctx_read * kvb + new_tokens * kvb
        return kv_math.roofline_time(flops, bytes_, gpu)[0]

    def attn(q: int, ctx: int) -> float:
        return kv_math.attention_flops(model, q, ctx)

    static_batch: list[Req] = []
    while waiting or running or static_batch:
        arrived = [r for r in waiting if r.arrival <= now]
        if not running and not arrived and not static_batch:
            now = waiting[0].arrival  # idle until next arrival
            continue

        if policy == "static":
            if not static_batch:
                take, budget = [], free
                for r in arrived[:max_batch]:
                    need = blocks_for(r.prompt + r.output)
                    if need <= budget:
                        take.append(r)
                        budget -= need
                if not take:
                    raise RuntimeError("request larger than the whole KV pool")
                for r in take:
                    waiting.remove(r)
                    r.blocks = blocks_for(r.prompt + r.output)  # reserve worst case
                    free -= r.blocks
                static_batch = take
                q = sum(r.prompt for r in take)
                now += step_time(q, 0, sum(attn(r.prompt, r.prompt // 2) for r in take))
                for r in take:
                    r.prefilled = r.prompt
                    r.generated = 1
                    r.first_token = now
                    r.token_times.append(now)
                iters += 1
                batch_sum += len(take)
                continue
            live = [r for r in static_batch if r.generated < r.output]
            if not live:
                for r in static_batch:
                    free += r.blocks
                static_batch = []
                continue
            # the whole batch occupies the GPU; finished rows are padding
            now += step_time(len(static_batch), sum(r.context for r in live),
                             sum(attn(1, r.context) for r in live))
            for r in live:
                r.generated += 1
                r.token_times.append(now)
                if r.generated >= r.output:
                    r.finish = now
            iters += 1
            batch_sum += len(static_batch)
            continue

        # ---- continuous / chunked -------------------------------------------------
        # 1. grow decodes: each decoding request may need a new block
        for r in list(running):
            if r.prefilled < r.prompt:
                continue
            if blocks_for(r.context + 1) > r.blocks:
                while free == 0:
                    victim = running[-1]  # most recently admitted
                    running.remove(victim)
                    free += victim.blocks
                    victim.blocks = 0
                    victim.prompt = victim.prompt + victim.generated  # recompute everything
                    victim.output -= victim.generated
                    victim.generated = 0
                    victim.prefilled = 0
                    victim.preemptions += 1
                    preempt += 1
                    waiting.insert(0, victim)
                    if victim is r:
                        break
                if r in running:
                    free -= 1
                    r.blocks += 1

        # 2. admit
        for r in [w for w in waiting if w.arrival <= now]:
            if len(running) >= max_batch:
                break
            # Admission control: the prompt's blocks are allocated at admission (both policies).
            # Allocating per chunk with no reservation lets many half-prefilled prompts hold
            # partial blocks and thrash under pressure.
            need = blocks_for(r.prompt + 1)
            if need > free:
                break  # FCFS: don't skip ahead
            waiting.remove(r)
            free -= need
            r.blocks = need
            running.append(r)

        if not running:
            if waiting:
                now = max(now, min(w.arrival for w in waiting))
            continue

        # 3. build the iteration
        prefills = [r for r in running if r.prefilled < r.prompt]
        decodes = [r for r in running if r.prefilled >= r.prompt]
        chunks: dict[int, int] = {}
        if policy == "continuous":
            if prefills:  # prefill-only iteration: decodes wait
                chunks = {r.rid: r.prompt - r.prefilled for r in prefills}
                decodes = []
        else:  # chunked
            budget = token_budget - len(decodes)
            for r in prefills:
                if budget <= 0:
                    break
                c = min(budget, r.prompt - r.prefilled)
                chunks[r.rid] = c
                budget -= c
        new_tokens = len(decodes) + sum(chunks.values())
        ctx_read = sum(r.context for r in decodes) + sum(
            r.prefilled for r in prefills if r.rid in chunks)
        flops = sum(attn(1, r.context) for r in decodes) + sum(
            attn(c, r.prefilled + c // 2) for r in prefills if (c := chunks.get(r.rid, 0)))
        now += step_time(new_tokens, ctx_read, flops)
        iters += 1
        batch_sum += len(decodes) + len(chunks)

        for r in decodes:
            r.generated += 1
            r.token_times.append(now)
        for r in prefills:
            c = chunks.get(r.rid, 0)
            r.prefilled += c
            if c and r.prefilled >= r.prompt:  # prefill completes -> first token
                r.generated += 1
                r.token_times.append(now)
                if r.first_token is None:
                    r.first_token = now
        for r in list(running):
            if r.prefilled >= r.prompt and r.generated >= r.output:
                r.finish = now
                running.remove(r)
                free += r.blocks
                r.blocks = 0

    ttft = [r.first_token - r.arrival for r in reqs if r.first_token is not None]
    tpot = [(r.finish - r.first_token) / max(1, len(r.token_times) - 1)
            for r in reqs if r.finish is not None and r.first_token is not None]
    itl = [b - a for r in reqs for a, b in zip(r.token_times, r.token_times[1:])]
    total_out = sum(len(r.token_times) for r in reqs)
    start = min(r.arrival for r in reqs)
    return Result(policy, now - start, total_out / (now - start), sum(ttft) / len(ttft),
                  _pct(ttft, 0.99), sum(tpot) / len(tpot), _pct(itl, 0.99), preempt,
                  batch_sum / max(1, iters), iters)
