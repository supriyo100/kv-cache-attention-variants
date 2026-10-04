"""Tests for inference_lab. Each test pins a claim the site makes in prose."""

from __future__ import annotations

import random

import pytest
import torch
import torch.nn.functional as F

from inference_lab import (
    disagg,
    eviction,
    flash,
    fragmentation,
    kv_math,
    mla,
    paged,
    prefix,
    quant,
    scheduler,
    specdec,
)

# ---------------------------------------------------------------- kv_math


def test_llama3_8b_kv_per_token_is_128_kib():
    # 32 layers * 2 * 8 kv heads * 128 dims * 2 bytes = 131072 bytes
    assert kv_math.kv_bytes_per_token(kv_math.LLAMA3_8B) == 131072


def test_mla_counts_latent_plus_rope_key_not_k_and_v():
    # DeepSeek-V3: 61 * (512 + 64) * 2 bytes = 70272 bytes per token
    assert kv_math.kv_bytes_per_token(kv_math.DEEPSEEK_V3) == 61 * 576 * 2


def test_variant_ordering():
    m = kv_math.LLAMA3_8B
    mha, gqa, mqa = (kv_math.kv_bytes_per_token_variant(m, v) for v in ("mha", "gqa", "mqa"))
    assert mha == 4 * gqa == 32 * mqa


def test_decode_is_memory_bound_and_prefill_compute_bound():
    m, g = kv_math.LLAMA3_8B, kv_math.H100_SXM
    _, bound_decode = kv_math.roofline_time(*kv_math.step_cost(m, 1, 1, 1024), g)
    _, bound_prefill = kv_math.roofline_time(*kv_math.step_cost(m, 1, 4096, 2048), g)
    assert bound_decode == "memory" and bound_prefill == "compute"


# ---------------------------------------------------------------- paged


def test_paged_random_workload_keeps_invariants():
    rng = random.Random(0)
    cache = paged.PagedKVCache(num_blocks=64, block_size=4)
    live: list[str] = []
    for i in range(400):
        op = rng.random()
        try:
            if op < 0.25 or not live:
                sid = f"s{i}"
                cache.add(sid, rng.randint(1, 9))
                live.append(sid)
            elif op < 0.35:
                child = f"c{i}"
                cache.fork(rng.choice(live), child)
                live.append(child)
            elif op < 0.5:
                sid = live.pop(rng.randrange(len(live)))
                cache.free(sid)
            else:
                cache.append_token(rng.choice(live))
        except paged.OutOfBlocks:
            sid = live.pop(0)
            cache.free(sid)
        cache.check_invariants()


def test_block_table_translation():
    c = paged.PagedKVCache(num_blocks=8, block_size=4)
    c.add("a", 3)
    c.add("b", 4)
    c.append_token("a")  # a fills block 0
    c.append_token("a")  # a needs a new block -> 2
    assert c.seqs["a"].table == [0, 2]
    assert c.physical_slot("a", 4) == (2, 0)


def test_copy_on_write_on_shared_partial_block():
    c = paged.PagedKVCache(num_blocks=8, block_size=4)
    c.add("p", 6)  # blocks [0, 1]; block 1 half full
    c.fork("p", "q")
    assert c.alloc.refcount[1] == 2
    c.append_token("q")  # writes into shared partial block -> copy
    assert c.alloc.copies == 1
    assert c.seqs["q"].table[0] == 0 and c.seqs["q"].table[1] != 1
    assert c.alloc.refcount[0] == 2 and c.alloc.refcount[1] == 1
    c.check_invariants()


def test_internal_waste_below_block_size():
    for n in range(1, 200):
        assert 0 <= fragmentation.internal_waste(n, 16) < 16


# ---------------------------------------------------------------- fragmentation


def test_paged_beats_contiguous_on_same_trace():
    trace = fragmentation.make_trace(300, seed=1)
    cont = fragmentation.simulate(trace, 64 * 1024, "contiguous")
    pag = fragmentation.simulate(trace, 64 * 1024, "paged")
    assert pag.mean_utilization > 0.9 > cont.mean_utilization
    assert pag.admitted >= cont.admitted


# ---------------------------------------------------------------- prefix


def test_hash_cache_shares_full_prefix_blocks_only():
    c = prefix.HashBlockCache(num_blocks=32, block_size=4)
    system = list(range(100, 112))  # 12 tokens = 3 full blocks
    t1, hit1 = c.acquire(system + [1, 2, 3, 4, 5])
    t2, hit2 = c.acquire(system + [9, 9, 9, 9])
    assert hit1 == 0 and hit2 == 12
    assert t1[:3] == t2[:3] and t1[3] != t2[3]


def test_hash_chain_position_matters():
    # same block content after different prefixes must not collide
    a = prefix.block_hashes([1, 2, 3, 4, 5, 6, 7, 8], 4)
    b = prefix.block_hashes([9, 9, 9, 9, 5, 6, 7, 8], 4)
    assert a[1] != b[1]


def test_released_blocks_stay_cached_until_evicted():
    c = prefix.HashBlockCache(num_blocks=4, block_size=2)
    t, _ = c.acquire([1, 2, 3, 4, 5, 6])
    c.release(t)
    _, hit = c.acquire([1, 2, 3, 4, 5, 6])
    assert hit == 4  # first two blocks come back from cache; the last block is recomputed


def test_lru_order_is_free_list_order():
    c = prefix.HashBlockCache(num_blocks=4, block_size=2)
    a, _ = c.acquire([1, 2, 3])  # blocks for [1,2] (hashed) and [3]
    c.release(a)  # tail block freed first, so the prefix block is the most recently freed
    b, _ = c.acquire([7, 8, 9, 9, 5])  # 3 blocks: takes the 3 least recently freed
    c.release(b)
    t, hit = c.acquire([1, 2, 3])
    assert hit == 2  # the prefix block survived
    c.release(t)
    b, _ = c.acquire([7, 8, 9, 9, 5, 5, 6])  # 4 blocks: every block is reallocated
    c.release(b)
    _, hit = c.acquire([1, 2, 3])
    assert hit == 0 and c.evictions >= 1


def test_hash_hit_verifies_tokens():
    c = prefix.HashBlockCache(num_blocks=8, block_size=2)
    t, _ = c.acquire([1, 2, 3])
    c.block_tokens[t[0]] = (9, 9)  # simulate a hash collision: stored tokens differ
    _, hit = c.acquire([1, 2, 3])
    assert hit == 0


def test_radix_longest_prefix_and_split():
    r = prefix.RadixCache(1000)
    hit, p1 = r.insert([1, 2, 3, 4, 5])
    assert hit == 0
    hit, p2 = r.insert([1, 2, 3, 9])
    assert hit == 3
    assert r.cached_tokens() == 6  # 1,2,3 shared + 4,5 + 9
    r.release(p1)
    r.release(p2)


def test_radix_evicts_unlocked_lru_leaves_only():
    r = prefix.RadixCache(8)
    _, a = r.insert([1, 2, 3, 4])
    _, _b = r.insert([5, 6, 7, 8])
    r.release(a)  # a is evictable, b is locked
    _, _c = r.insert([9, 10])
    assert r.match([5, 6, 7, 8])[0] == 4
    assert r.match([1, 2, 3, 4])[0] < 4
    with pytest.raises(RuntimeError):
        r.insert(list(range(20, 40)))  # needs more than the unlocked space


# ---------------------------------------------------------------- scheduler


def test_continuous_beats_static_throughput():
    trace = scheduler.poisson_trace(120, rate=20.0, seed=3)
    s = scheduler.simulate(trace, "static")
    c = scheduler.simulate(trace, "continuous")
    assert c.throughput_tok_s > s.throughput_tok_s
    assert c.ttft_mean < s.ttft_mean


def test_chunked_prefill_bounds_inter_token_latency():
    trace = scheduler.poisson_trace(120, rate=20.0, seed=3, prompt=(1024, 4096))
    c = scheduler.simulate(trace, "continuous")
    k = scheduler.simulate(trace, "chunked")
    assert k.itl_p99 < c.itl_p99


def test_preemption_under_memory_pressure_still_completes():
    trace = scheduler.poisson_trace(60, rate=50.0, seed=0)
    r = scheduler.simulate(trace, "chunked", num_blocks=300)
    assert r.preemptions > 0
    assert r.throughput_tok_s > 0


# ---------------------------------------------------------------- flash


@pytest.mark.parametrize("causal", [False, True])
def test_tiled_matches_sdpa(causal):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 70, 32, dtype=torch.float64) for _ in range(3))
    o, _ = flash.tiled_attention(q, k, v, 16, 16, causal=causal)
    torch.testing.assert_close(o, F.scaled_dot_product_attention(q, k, v, is_causal=causal))
    torch.testing.assert_close(o, flash.naive_attention(q, k, v, causal=causal))


def test_split_kv_merge_is_exact():
    torch.manual_seed(1)
    q = torch.randn(1, 8, 1, 64, dtype=torch.float64)
    k, v = torch.randn(1, 8, 500, 64, dtype=torch.float64), torch.randn(1, 8, 500, 64, dtype=torch.float64)
    torch.testing.assert_close(flash.split_kv_attention(q, k, v, 7), F.scaled_dot_product_attention(q, k, v))


def test_flash_io_ratio_matches_theory():
    # Saving ~ M / (2 d^2) for large N (FlashAttention Thm 2).
    n, m = 8192, 100_000
    for d in (64, 128):
        ratio = flash.io_naive(n, d) / flash.io_flash(n, d, m)
        assert ratio == pytest.approx(m / (2 * d * d), rel=0.25)


# ---------------------------------------------------------------- mla (on the course module)


def _course_mla():
    from kv_cache_variants.attention.mla import MultiHeadLatentAttention
    from kv_cache_variants.rope import build_rope_cache

    torch.manual_seed(2)
    m = MultiHeadLatentAttention(d_model=64, num_heads=4, latent_dim=16, rope_head_dim=8).double()
    return m, tuple(t.double() for t in build_rope_cache(8, max_seq_len=64))


@pytest.mark.parametrize("fold", [False, True])
def test_mla_absorbed_equals_course_module(fold):
    m, rope = _course_mla()
    x = torch.randn(2, 10, 64, dtype=torch.float64)
    ref, cache = m(x, rope=rope)  # prefill, explicit
    out, cache2 = mla.absorbed_forward(m, x, rope=rope, fold_output=fold)
    torch.testing.assert_close(out, ref)
    step = torch.randn(2, 1, 64, dtype=torch.float64)
    ref1, _ = m(step, past_kv=cache, rope=rope)  # decode, explicit
    out1, _ = mla.absorbed_forward(m, step, past_kv=cache2, rope=rope, fold_output=fold)
    torch.testing.assert_close(out1, ref1)


def test_mla_absorbed_decode_needs_fewer_flops_at_long_context():
    m, _ = _course_mla()
    assert mla.decode_flops(m, 4096, absorbed=True) < mla.decode_flops(m, 4096, absorbed=False) / 4


def test_mla_cache_is_smaller_than_mha():
    assert mla.cache_elems_per_token(512, 64) * 28 < mla.equivalent_mha_elems_per_token(128, 128)


# ---------------------------------------------------------------- quant


def test_per_channel_keys_beat_per_token_with_outliers():
    k, _ = quant.synthetic_kv()
    e_tok = quant.rel_error(quant.quantize_grouped(k, 2, "per_token"), k)
    e_ch = quant.rel_error(quant.quantize_grouped(k, 2, "per_channel"), k)
    assert e_ch < e_tok


def test_kivi_keeps_residual_exact():
    k, v = quant.synthetic_kv(t=100)
    kh, _ = quant.kivi(k, v, bits=2, group=32, residual=32)
    torch.testing.assert_close(kh[64:], k[64:])


def test_lloyd_max_matches_known_gaussian_codebook():
    # Classical Lloyd-Max levels for N(0,1): 1 bit +-0.798; 2 bits +-0.453, +-1.510.
    torch.testing.assert_close(quant.lloyd_max_gaussian(1).abs(), torch.tensor([0.798, 0.798]), atol=5e-3, rtol=0)
    torch.testing.assert_close(quant.lloyd_max_gaussian(2).abs().sort().values,
                               torch.tensor([0.453, 0.453, 1.510, 1.510]), atol=1e-2, rtol=0)


def test_rotation_removes_outlier_penalty():
    k, _ = quant.synthetic_kv()
    k_plain, _ = quant.synthetic_kv(outlier_scale=0.0)
    e_tq = quant.rel_error(quant.turboquant(k, 3), k)
    e_tq_plain = quant.rel_error(quant.turboquant(k_plain, 3), k_plain)
    e_tok = quant.rel_error(quant.quantize_grouped(k, 3, "per_token"), k)
    assert e_tq < e_tok
    assert e_tq < 0.25 and e_tq_plain < 0.25  # rotation makes error insensitive to outliers


# ---------------------------------------------------------------- eviction


def test_snapkv_and_h2o_beat_streaming_on_needles():
    k, v, d, _ = eviction.needle_context()
    q_obs = eviction.queries_toward(d, 32, seed=5)
    q_future = eviction.queries_toward(d, 16, seed=6)
    budget = 128
    e_s = eviction.output_error(q_future, k, v, eviction.streaming(k.shape[0], budget))
    keep_h2o = eviction.h2o(eviction.attention_probs(q_obs, k), budget, recent=64)
    assert eviction.output_error(q_future, k, v, eviction.snapkv(q_obs, k, budget)) < e_s
    assert eviction.output_error(q_future, k, v, keep_h2o) < e_s


# ---------------------------------------------------------------- specdec


def test_speculative_sampling_preserves_target_distribution():
    gen = torch.Generator().manual_seed(0)
    p = torch.tensor([0.5, 0.3, 0.15, 0.05])
    q = torch.full((4,), 0.25)
    counts = torch.zeros(4)
    n = 40_000
    for _ in range(n):
        x, _ = specdec.speculative_step(p, q, gen)
        counts[x] += 1
    torch.testing.assert_close(counts / n, p, atol=0.012, rtol=0)


def test_expected_tokens_formula_matches_monte_carlo():
    for alpha, gamma in [(0.6, 4), (0.8, 6), (0.9, 3)]:
        assert abs(specdec.simulate_chain(alpha, gamma, 200_000) - specdec.expected_tokens(alpha, gamma)) < 0.03


def test_acceptance_rate_is_one_minus_tv():
    p, q = torch.tensor([0.7, 0.2, 0.1]), torch.tensor([0.5, 0.3, 0.2])
    assert specdec.acceptance_rate(p, q) == pytest.approx(1 - 0.5 * float((p - q).abs().sum()))


def test_confidence_schedule_beats_fixed_length_under_budget():
    gen = torch.Generator().manual_seed(3)
    alphas = torch.rand(32, 8, generator=gen) * 0.5 + 0.45  # per-request confidence varies
    lengths, total = specdec.confidence_schedule(alphas, slot_budget=32 * 4)
    _k, total_fixed = specdec.fixed_schedule(alphas, slot_budget=32 * 4)
    assert int(lengths.sum()) <= 32 * 3 and total >= total_fixed
    assert specdec.expected_tokens_positional(torch.full((4,), 0.8))[-1] == pytest.approx(
        specdec.expected_tokens(0.8, 4))


# ---------------------------------------------------------------- disagg


def test_kv_transfer_small_vs_prefill_on_nvlink():
    r = disagg.transfer_ratio(kv_math.LLAMA3_70B, kv_math.H100_SXM, 8192, 450.0)
    assert r < 0.1
