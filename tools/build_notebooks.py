"""Generate the original notebooks in notebooks/ (executing them is optional).

Each notebook follows the same sections: objective, theory, equations, diagram,
implementation, experiment, results, interpretation, limitations, conclusion. Code cells
import ``inference_lab`` so the notebooks stay thin and every result comes from tested code.

    uv run python tools/build_notebooks.py                      # write all, no execution
    uv run python tools/build_notebooks.py --execute sota_02    # also run (needs a capable machine)
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
REPO = "https://github.com/supriyo100/kv-cache-attention-variants"

SETUP = f"""# Setup: works locally (uv sync) and on Colab (installs the small inference_lab package only)
try:
    import inference_lab  # noqa: F401
except ImportError:  # Colab / fresh environment
    %pip install -q --no-deps "git+{REPO}"
import torch
import matplotlib.pyplot as plt
plt.rcParams.update({{"figure.figsize": (7.5, 3.6), "axes.spines.top": False, "axes.spines.right": False}})
C = {{"compute": "#0E7C72", "memory": "#B5790A", "waste": "#B42B3E", "shared": "#5B47C9", "ink": "#1C232B"}}"""


def md(s: str):
    return nbformat.v4.new_markdown_cell(s.strip("\n"))


def code(s: str):
    return nbformat.v4.new_code_cell(s.strip("\n"))


def header(title: str, step: str, sources: str) -> list:
    return [md(f"""# {title}

Part of **[Inference Engineering]({REPO})**, an original notebook (not from the upstream course).
Derivation: {step}. Sources: {sources}.

Every number below is produced by code in this notebook or in `inference_lab` (tested in
`tests/`). Labels: *measured* = ran here (CPU), *theoretical* = formula with datasheet inputs."""),
            code(SETUP)]


NOTEBOOKS: dict[str, list] = {}

# ---------------------------------------------------------------------------------- 05
NOTEBOOKS["05_kv_cache_memory_calculator"] = header(
    "05 · KV cache memory calculator", "path step 4", "model config.json files; NVIDIA datasheets") + [
    md("""## Objective
Compute KV cache size for real models, find how many sequences fit on a GPU, and see how the
attention variant and the KV dtype change the answer."""),
    md(r"""## Theory and equations
Per token, a standard attention layer caches one key and one value per KV head:

$$m_{\text{tok}} = L \cdot 2 \cdot H_{kv} \cdot d_h \cdot s, \qquad M_{\text{KV}} = m_{\text{tok}}\sum_i T_i.$$

MLA caches a latent plus a shared RoPE key, with no separate V: $m_{\text{tok}} = L (d_c + d_R) s$."""),
    md("""## Diagram
```
HBM (80 GB) = [ weights 16 GB ][ reserve 8 GB ][ KV cache: 56 GB = 427,246 tokens of Llama-3-8B BF16 ]
```"""),
    md("## Implementation"),
    code("""from inference_lab import kv_math as K
for m in K.MODELS.values():
    b = K.kv_bytes_per_token(m)
    print(f"{m.name:12s} {m.attention:4s}  {b/1024:7.1f} KiB/token   {b*32768/2**30:6.2f} GiB per 32K sequence")"""),
    md("## Experiment: capacity on an H100 vs context length and KV dtype (*theoretical*)"),
    code("""ctx = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
fig, ax = plt.subplots()
for dtype, col in [("bf16", C["memory"]), ("fp8", C["compute"])]:
    n = K.max_concurrent_tokens(K.LLAMA3_8B, K.H100_SXM, kv_dtype=dtype)
    seqs = [n // c for c in ctx]
    ax.plot(ctx, seqs, "o-", color=col, label=f"Llama-3-8B, KV {dtype}")
    print(dtype, dict(zip(ctx, seqs)))
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("context per sequence (tokens)"); ax.set_ylabel("concurrent sequences on 1 x H100")
ax.legend(); plt.show()"""),
    md("## Results: the same model under four attention variants (*theoretical*)"),
    code("""for v in ("mha", "gqa", "mqa", "mla"):
    b = K.kv_bytes_per_token_variant(K.LLAMA3_8B, v)
    print(f"{v:4s} {b/1024:6.1f} KiB/token  ->  {int(56e9 // (b*32768)):4d} x 32K sequences in 56 GB")"""),
    md("""## Interpretation
Capacity falls as $1/T$: every doubling of context halves concurrency. FP8 KV doubles it back.
The variant matters more than the dtype: MHA → GQA-4 is 4×, GQA → MQA another 8×.

## Limitations
Ignores fragmentation (step 6), activation peaks beyond the flat 10% reserve, and per-GPU
replication under tensor parallelism.

## Conclusion
At long context, serving is capacity-bound before it is compute-bound. The cheapest levers are
the KV dtype and choosing a model with an efficient attention variant."""),
]

# ---------------------------------------------------------------------------------- 07
NOTEBOOKS["07_external_internal_fragmentation"] = header(
    "07 · External and internal fragmentation", "path step 6", "Kwon et al., SOSP 2023") + [
    md("""## Objective
Measure reservation waste, external fragmentation and internal fragmentation by replaying one
request trace through a contiguous allocator and a paged allocator."""),
    md(r"""## Theory and equations
* Reservation: a contiguous allocator reserves $T_p + N_{\max}$; mean use is $(T_p + N/2)/(T_p+N_{\max})$.
* External: free memory exists, but no hole is large enough.
* Internal: $w(T) = (-T) \bmod B < B$, mean $(B-1)/2$ if lengths are uniform mod $B$."""),
    md("""## Diagram
```
contiguous: [A....reserved....][hole][B..reserved..][ hole ][C......reserved......]
paged:      [A][A][a.][B][B][B][b..][C][c...][free][free][free]   (lower-case = partial last block)
```"""),
    md("## Implementation"),
    code("""from inference_lab import fragmentation as F
trace = F.make_trace(300, seed=1)
print(trace[:3])"""),
    md("## Experiment (*measured*, deterministic CPU simulation)"),
    code("""res = {m: F.simulate(trace, 64 * 1024, m) for m in ("contiguous", "paged")}
for m, s in res.items():
    print(f"{m:10s} admitted={s.admitted:3d}  rejected_external={s.rejected_external:3d}  "
          f"rejected_full={s.rejected_full:3d}  utilization={s.mean_utilization:.1%}  "
          f"waste={s.mean_reservation_waste + s.mean_internal_waste:,.0f} slots")"""),
    md("## Results: internal waste vs block size"),
    code("""bs = [1, 2, 4, 8, 16, 32, 64, 128]
util = [F.simulate(trace, 64 * 1024, "paged", block_size=b).mean_utilization for b in bs]
fig, ax = plt.subplots()
ax.plot(bs, [100 * (1 - u) for u in util], "o-", color=C["waste"])
ax.set_xscale("log", base=2); ax.set_xlabel("block size (tokens)"); ax.set_ylabel("internal waste (% of allocated)")
plt.show()
print([f"{b}:{100*(1-u):.2f}%" for b, u in zip(bs, util)])"""),
    md("""## Interpretation
The contiguous allocator rejects more than half the trace while memory is free in total. That
is external fragmentation. Paging removes it and turns reservation waste into a sub-block
remainder. Internal waste grows roughly linearly with block size and stays small up to 16–32.

## Limitations
No preemption (paged mode stops a request early if the pool is dry) and no compaction in the
contiguous allocator. The trace is synthetic.

## Conclusion
Paging fixes the waste that scales with sequence length. What remains is bounded by the block
size."""),
]

# ---------------------------------------------------------------------------------- 08
NOTEBOOKS["08_paged_kv_allocator_simulation"] = header(
    "08 · Paged KV allocator: block tables, reference counts, copy-on-write", "path steps 7–8",
    "Kwon et al., SOSP 2023; github.com/vllm-project/vllm") + [
    md("""## Objective
Drive `inference_lab.paged` by hand, watch block tables form, translate token positions to
physical slots, and trigger copy-on-write."""),
    md(r"""## Theory and equations
$\text{slot}(r,p) = \text{table}_r[\lfloor p/B\rfloor]\cdot B + (p \bmod B)$, and
$\text{ref}(b) = |\{(r,i): \text{table}_r[i] = b\}|$. A block returns to the free list when its refcount reaches 0."""),
    md("""## Diagram
```
seq "a": logical [0][1][2]  -> table [0, 2, 5]      physical pool: 0:a 1:b 2:a 3:b 4:- 5:a ...
```"""),
    md("## Implementation"),
    code("""from inference_lab.paged import PagedKVCache, OutOfBlocks
c = PagedKVCache(num_blocks=12, block_size=4)
c.add("a", 5); c.add("b", 7)
for _ in range(4): c.append_token("a")
print("tables:", {k: v.table for k, v in c.seqs.items()}, "free:", c.alloc.num_free)
print("token 8 of a lives at (block, offset) =", c.physical_slot("a", 8))
c.check_invariants()"""),
    md("## Experiment: fork and copy-on-write (*measured*)"),
    code("""c.fork("b", "b2")                           # parallel sample shares all of b's blocks
print("refcounts after fork:", c.alloc.refcount)
c.append_token("b2")                         # writes into the shared, partially filled last block
print("copy-on-write events:", c.alloc.copies, " b:", c.seqs["b"].table, " b2:", c.seqs["b2"].table)
c.free("b"); print("after freeing b, refcounts:", c.alloc.refcount)
c.check_invariants()"""),
    md("## Results: running out of blocks"),
    code("""try:
    while True:
        c.append_token("a")
except OutOfBlocks as e:
    print("OutOfBlocks:", e, "| a holds", c.seqs["a"].tokens, "tokens in", len(c.seqs["a"].table), "blocks")"""),
    md("""## Interpretation
Physical placement is arbitrary; the table preserves order. Sharing is a refcount increment,
and the first divergent write pays one block copy. Exhaustion surfaces as an exception the
scheduler must handle (preempt, swap, or queue).

## Limitations
No GPU memory; the pool is a Python list. Real engines batch these operations per iteration.

## Conclusion
A paged KV manager is a small amount of bookkeeping with strong invariants. The tests check
those invariants after hundreds of random operations."""),
]

# ---------------------------------------------------------------------------------- 11
NOTEBOOKS["11_continuous_batching_simulation"] = header(
    "11 · Static vs continuous batching vs chunked prefill", "path step 9",
    "Orca (OSDI 2022); Sarathi-Serve (arXiv:2403.02310)") + [
    md("""## Objective
Compare scheduling policies on the same Poisson trace: throughput, TTFT, TPOT, p99 ITL and
preemptions."""),
    md(r"""## Theory and equations
$\mathcal{R}_{t+1} = (\mathcal{R}_t \setminus \mathcal{F}_t) \cup \mathcal{A}_t$. Chunked prefill caps
the tokens per iteration at $\tau$, so iteration time, and therefore ITL, is bounded. Each forward
pass is timed with the roofline model: $t = \max(F/\pi, Q/\beta)$."""),
    md("""## Diagram
```
static:      [prefill A..F][decode ......... until the longest finishes][prefill G..L] ...
continuous:  A B C D E F | F done -> G admitted next iteration | ...
chunked:     every iteration = all decodes + up to (tau - decodes) prompt tokens
```"""),
    md("## Implementation"),
    code("""from inference_lab import scheduler as S
trace = S.poisson_trace(200, rate=20.0, seed=3)"""),
    md("## Experiment (*theoretical*: roofline-timed, Llama-3-8B on one H100)"),
    code("""rows = [S.simulate(trace, p) for p in ("static", "continuous", "chunked")]
print(f"{'policy':11s} {'tok/s':>7s} {'TTFT':>8s} {'TTFT p99':>9s} {'TPOT':>8s} {'ITL p99':>8s} {'preempt':>8s}")
for r in rows:
    print(f"{r.policy:11s} {r.throughput_tok_s:7.0f} {r.ttft_mean:7.3f}s {r.ttft_p99:8.3f}s "
          f"{1e3*r.tpot_mean:6.1f}ms {1e3*r.itl_p99:6.1f}ms {r.preemptions:8d}")"""),
    md("## Results: load sweep"),
    code("""rates = [5, 10, 15, 20, 25, 30]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for p, col in [("static", C["waste"]), ("continuous", C["memory"]), ("chunked", C["compute"])]:
    rs = [S.simulate(S.poisson_trace(150, rate=r, seed=1), p) for r in rates]
    ax[0].plot(rates, [x.ttft_mean for x in rs], "o-", color=col, label=p)
    ax[1].plot(rates, [1e3 * x.itl_p99 for x in rs], "o-", color=col, label=p)
ax[0].set_xlabel("arrival rate (req/s)"); ax[0].set_ylabel("mean TTFT (s)"); ax[0].set_yscale("log")
ax[1].set_xlabel("arrival rate (req/s)"); ax[1].set_ylabel("p99 ITL (ms)"); ax[1].legend()
plt.tight_layout(); plt.show()"""),
    md("""## Interpretation
Static batching has smooth decode but queues requests behind whole batches. Prefill-first
continuous batching fixes TTFT and throughput, and creates ITL spikes. Chunked prefill keeps
the TTFT gains and bounds ITL.

## Limitations
Roofline timing is an upper bound on speed. No CPU scheduling overhead, no CUDA graphs, no
prefix caching. Static mode reserves each request's true length, which is generous to it.

## Conclusion
The scheduling policy moves latency percentiles by an order of magnitude on identical hardware."""),
]

# ---------------------------------------------------------------------------------- 12
NOTEBOOKS["12_prefill_vs_decode"] = header(
    "12 · Prefill vs decode on the roofline", "path step 2",
    "Williams et al., Roofline (CACM 2009); NVIDIA H100 datasheet") + [
    md("""## Objective
Compute arithmetic intensity and roofline step time for prefill and decode, and find the
batch-size ceiling set by KV reads."""),
    md(r"""## Theory and equations
$I_{\text{prefill}} \approx 2T/s$; $I_{\text{decode}} \approx \dfrac{2PB}{sP + BTm_{\text{tok}}} \to \dfrac{2P}{Tm_{\text{tok}}}$ as $B\to\infty$."""),
    md("""## Diagram
```
TFLOP/s |            ______________ 989 (compute roof)
        |          /  ^ prefill
        |        /
        |  B=256 * <- KV ceiling
        |   B=1 *     (bandwidth slope 3.35 TB/s)
        +------------------------------ FLOP/byte (ridge = 295)
```"""),
    md("## Implementation"),
    code("""from inference_lab import kv_math as K
m, g = K.LLAMA3_8B, K.H100_SXM
print("ridge point:", round(g.ridge_bf16), "FLOP/byte")"""),
    md("## Experiment (*theoretical*)"),
    code("""for B in (1, 8, 32, 128, 256, 1024):
    f, q = K.step_cost(m, B, 1, 2048); t, bound = K.roofline_time(f, q, g)
    print(f"decode B={B:5d}  I={f/q:6.1f}  step={1e3*t:6.2f} ms  {bound:7s}  {B/t:8.0f} tok/s")
for T in (128, 512, 2048, 8192):
    f, q = K.step_cost(m, 1, T, T // 2); t, bound = K.roofline_time(f, q, g)
    print(f"prefill T={T:5d} I={f/q:7.0f}  time={1e3*t:6.2f} ms  {bound}")"""),
    md("## Results: the KV ceiling vs context"),
    code("""ctx = [512, 2048, 8192, 32768]
B = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
fig, ax = plt.subplots()
for T, a in zip(ctx, [1, .75, .5, .3]):
    I = [K.arithmetic_intensity(*K.step_cost(m, b, 1, T)) for b in B]
    ax.plot(B, I, "o-", color=C["memory"], alpha=a, label=f"T={T}")
ax.axhline(g.ridge_bf16, color=C["ink"], ls="--", lw=1); ax.text(1, g.ridge_bf16 * 1.1, "H100 ridge")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("batch"); ax.set_ylabel("decode FLOP/byte")
ax.legend(); plt.show()"""),
    md("""## Interpretation
Decode intensity saturates far below the ridge, and lower at longer contexts. Past the knee,
batching adds latency without adding much throughput.

## Limitations
First-order model: ignores activations, kernel efficiency, and launch overhead.

## Conclusion
Decode is a bandwidth problem; make bytes smaller (quantization, GQA/MLA) or produce more
tokens per byte (speculative decoding)."""),
]

# ---------------------------------------------------------------------------------- 13
NOTEBOOKS["13_flash_attention_io_model"] = header(
    "13 · FlashAttention: online softmax and the IO model", "path step 10",
    "Dao et al. (arXiv:2205.14135); Milakov & Gimelshein (arXiv:1805.02867)") + [
    md("""## Objective
Show that tiled attention with an online softmax is exact, and count the HBM traffic it saves."""),
    md(r"""## Theory and equations
$m^{(j)} = \max(m^{(j-1)}, \operatorname{rowmax} S_j)$, $\ell^{(j)} = e^{m^{(j-1)}-m^{(j)}}\ell^{(j-1)} + \operatorname{rowsum} e^{S_j - m^{(j)}}$,
$\tilde o^{(j)} = e^{m^{(j-1)}-m^{(j)}}\tilde o^{(j-1)} + e^{S_j - m^{(j)}} V_j$, and $O = \tilde o/\ell$.
IO: naive $4Td + 4T^2$; tiled $\Theta(T^2d^2/M)$."""),
    md("""## Diagram
```
for each Q tile (in SRAM):            for each K,V tile (streamed):  S_tile -> update (m, l, o)
write O tile once                     the T x T matrix is never written to HBM
```"""),
    md("## Implementation"),
    code("""from inference_lab import flash
import torch.nn.functional as F
torch.manual_seed(0)
q, k, v = (torch.randn(1, 4, 300, 64) for _ in range(3))      # (batch, heads, seq, head_dim)
for causal in (False, True):
    o, _ = flash.tiled_attention(q, k, v, 32, 32, causal=causal)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    print("causal" if causal else "full  ", "max |tiled - SDPA| =", float((o - ref).abs().max()))"""),
    md("## Experiment: HBM traffic (*theoretical*, FP16, M = 100k elements of SRAM)"),
    code("""Ts = [512, 1024, 2048, 4096, 8192, 16384, 32768]
fig, ax = plt.subplots()
for d, a in [(64, .6), (128, 1)]:
    ax.plot(Ts, [2 * flash.io_naive(T, d) / 2**20 for T in Ts], "o-", color=C["waste"], alpha=a, label=f"naive d={d}")
    ax.plot(Ts, [2 * flash.io_flash(T, d, 100_000) / 2**20 for T in Ts], "o-", color=C["compute"], alpha=a, label=f"tiled d={d}")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("T"); ax.set_ylabel("HBM traffic per head (MiB)"); ax.legend(); plt.show()
for d in (64, 128):
    print(f"d={d}: traffic ratio at T=16384 = {flash.io_naive(16384, d) / flash.io_flash(16384, d, 100_000):.1f}x  (M/2d^2 = {100_000/(2*d*d):.1f})")"""),
    md("## Results: extra memory"),
    code("""for T in (4096, 32768, 131072):
    print(f"T={T:6d}: naive S+P per head = {2 * T * T * 2 / 2**30:8.2f} GiB   tiled extra = {T * 4 / 2**10:6.0f} KiB (LSE)")"""),
    md("""## Interpretation
The result is exact. By this count the traffic saving is $\\approx M/2d^2$: a few times at
$d=128$. The decisive change is memory, $O(T^2)$ to $O(T)$, and fusing three kernels into one.

## Limitations
PyTorch loops, not a fused kernel: no SRAM, warps or tensor cores. Real speedups depend on the kernel.

## Conclusion
FlashAttention is an IO and memory result. It does not change the size of the KV cache."""),
]

# ---------------------------------------------------------------------------------- SOTA
NOTEBOOKS["sota_01_radix_prefix_cache"] = header(
    "S01 · RadixAttention vs hash-chained prefix caching", "path step 8",
    "SGLang (arXiv:2312.07104, github.com/sgl-project/sglang); vLLM automatic prefix caching") + [
    md("""## Objective
Rebuild both prefix-cache designs and compare hit rates and memory on shared-prompt and
multi-turn workloads. Official code: `sglang/srt/mem_cache/radix_cache.py`,
`vllm/v1/core/kv_cache_manager.py`."""),
    md(r"""## Theory and equations
Hash chain: $h_i = H(h_{i-1}, x_{iB..(i+1)B})$, so only full blocks with identical prefixes match.
Radix tree: longest-prefix match at token granularity; lock counts; LRU leaf eviction."""),
    md("""## Diagram
```
root --[system prompt 2000, lock 3]--+--[A: q1]--[A: r1+q2]
                                     +--[B: q]
                                     +--[C: q, lock 0 -> evictable]
```"""),
    md("## Implementation"),
    code("""import random
from inference_lab.prefix import HashBlockCache, RadixCache
rng = random.Random(0)
system = list(range(10_000, 12_000))
users = [system + [rng.randint(0, 9999) for _ in range(200)] for _ in range(50)]"""),
    md("## Experiment 1: shared system prompt (*measured*, CPU)"),
    code("""hc = HashBlockCache(20_000, 16); rc = RadixCache(10**7)
h_hits = sum(hc.acquire(u)[1] for u in users)
r_hits = sum(rc.insert(u)[0] for u in users)
total = sum(map(len, users))
print(f"prompt tokens {total:,}; hash hits {h_hits:,} ({h_hits/total:.1%}); radix hits {r_hits:,} ({r_hits/total:.1%})")
print(f"hash blocks used {hc.capacity - len(hc.free)} vs {sum(-(-len(u)//16) for u in users)} without sharing")
print(f"radix cached tokens {rc.cached_tokens():,} vs {total:,} without sharing")"""),
    md("## Experiment 2: multi-turn chat and eviction pressure (*measured*)"),
    code("""def multiturn(capacity_tokens):
    rng = random.Random(1); rc = RadixCache(capacity_tokens); hits = tot = 0
    convs = [list(system[:500]) for _ in range(20)]
    for turn in range(6):
        for c in convs:
            c += [rng.randint(0, 9999) for _ in range(150)]
            h, path = rc.insert(list(c)); hits += h; tot += len(c); rc.release(path)
            c += [rng.randint(0, 9999) for _ in range(250)]
    return hits / tot, rc.evicted_tokens
caps = [5_000, 10_000, 20_000, 40_000, 80_000]
res = [multiturn(c) for c in caps]
for c, (hr, ev) in zip(caps, res): print(f"capacity {c:6,} tokens: hit rate {hr:.1%}, evicted {ev:,}")
plt.plot(caps, [100 * r[0] for r in res], "o-", color=C["shared"]); plt.xscale("log")
plt.xlabel("cache capacity (tokens)"); plt.ylabel("prompt tokens served from cache (%)"); plt.show()"""),
    md("""## Results and interpretation
Both designs reuse the shared prompt fully. The radix tree also reuses partial blocks.
Hit rate collapses once the hot working set exceeds capacity: LRU starts evicting
conversations that are about to send their next turn.

## Limitations
Generated replies are not inserted into the cache here (SGLang does insert them), so
multi-turn hit rates are a lower bound. No GPU memory or kernels.

## Conclusion
Prefix caching is a data-structure problem; the policy question is eviction under pressure."""),
]

NOTEBOOKS["sota_02_mla_absorption"] = header(
    "S02 · MLA weight absorption on the course's MLA module (the idea behind FlashMLA)", "path step 5",
    "DeepSeek-V2 (arXiv:2405.04434); github.com/deepseek-ai/FlashMLA (13k stars, MIT)") + [
    md("""## Objective
Take the course's `MultiHeadLatentAttention` (which up-projects its whole latent cache to K and V
at every step) and compute the *same* outputs in absorbed form from the same weights, never
building K or V. Then count what absorption saves."""),
    md(r"""## Theory and equations
$q_i^\top W_{UK}^i c_t = (W_{UK}^{i\top} q_i)^\top c_t$ and $o_i = W_{UV}^i \sum_t p_t c_t$; optionally fold $W_O^i W_{UV}^i$.
The cache stays $c_t \in \mathbb{R}^{d_c}$ plus the shared RoPE key $k^R_t \in \mathbb{R}^{d_R}$."""),
    md("""## Diagram
```
course module (explicit):  latent (T x d_c) --k_up_proj--> K (H x T x d_h) --v_up_proj--> V ... SDPA
absorbed (this notebook):  q_i --W_UK^T--> q~_i (d_c) . latent -> softmax -> attend latent -> W_UV (or W_O W_UV)
```"""),
    md("## Implementation"),
    code("""from kv_cache_variants.attention.mla import MultiHeadLatentAttention
from kv_cache_variants.rope import build_rope_cache
from inference_lab import mla

torch.manual_seed(0)
m = MultiHeadLatentAttention(d_model=256, num_heads=8, latent_dim=64, rope_head_dim=16)
rope = build_rope_cache(16, max_seq_len=16384)
x = torch.randn(1, 32, 256)
with torch.no_grad():
    ref, cache = m(x, rope=rope)                                # the course's forward (explicit)
    out, cache2 = mla.absorbed_forward(m, x, rope=rope)         # absorbed, same weights
    step = torch.randn(1, 1, 256)
    ref1, _ = m(step, past_kv=cache, rope=rope)
    out1, _ = mla.absorbed_forward(m, step, past_kv=cache2, rope=rope, fold_output=True)
print("prefill max |diff|:", float((out - ref).abs().max()), " decode max |diff|:", float((out1 - ref1).abs().max()))"""),
    md("## Experiment: decode FLOPs vs context (*theoretical*, from the module's shapes)"),
    code("""Ts = [256, 1024, 4096, 16384, 65536]
fe = [mla.decode_flops(m, T, absorbed=False) for T in Ts]
fa = [mla.decode_flops(m, T, absorbed=True) for T in Ts]
for T, a, b in zip(Ts, fe, fa):
    print(f"T={T:6d}  explicit {a/1e6:9.1f} MFLOP   absorbed {b/1e6:7.1f} MFLOP   {a/b:5.1f}x")
plt.loglog(Ts, fe, "o-", color=C["waste"], label="explicit: up-project the cache")
plt.loglog(Ts, fa, "o-", color=C["compute"], label="absorbed")
plt.xlabel("context T"); plt.ylabel("attention FLOPs per decode step"); plt.legend(); plt.show()"""),
    md("## Optional: wall-clock on your machine (*measured* only when you run it)"),
    code("""import time
def bench(f, T, reps=7):
    c = (torch.randn(1, T, 64), torch.randn(1, T, 16))
    best = float("inf")
    with torch.no_grad():
        f(step, c)
        for _ in range(reps):
            t = time.perf_counter(); f(step, c); best = min(best, time.perf_counter() - t)
    return best
for T in (1024, 8192):
    te = bench(lambda s, c: m(s, past_kv=c, rope=rope), T)
    ta = bench(lambda s, c: mla.absorbed_forward(m, s, past_kv=c, rope=rope), T)
    print(f"T={T}: explicit {1e3*te:.2f} ms, absorbed {1e3*ta:.2f} ms")"""),
    md("""## Interpretation
Both forms agree to float precision. The explicit form's cost grows with $T \\cdot H \\cdot d_h
\\cdot d_c$ (re-up-projecting the cache every step); the absorbed form's with $T \\cdot H \\cdot d_c$.

## Limitations
CPU PyTorch. On GPU the absorbed decode becomes compute-bound, which kernels like FlashMLA target.

## Conclusion
Decompression moves onto the single query. It is never applied to the cache."""),
]

NOTEBOOKS["sota_03_flash_decoding_lse_merge"] = header(
    "S03 · Split-KV decoding and the LSE merge (FlashDecoding, cascade attention)", "path step 10",
    "github.com/Dao-AILab/flash-attention (25k stars, BSD-3); FlashInfer cascade (arXiv:2501.01005)") + [
    md("""## Objective
Show that attention over disjoint KV chunks merges exactly, and use the merge for cascade
(shared-prefix) attention."""),
    md(r"""## Theory and equations
$\text{LSE} = \log(e^{\text{LSE}_A} + e^{\text{LSE}_B})$, $o = e^{\text{LSE}_A-\text{LSE}} o_A + e^{\text{LSE}_B-\text{LSE}} o_B$."""),
    md("""## Diagram
```
KV:  [ chunk 0 ][ chunk 1 ][ chunk 2 ][ chunk 3 ]   -> processed in parallel
      (o0,l0)    (o1,l1)    (o2,l2)    (o3,l3)       -> merged pairwise, exactly
```"""),
    md("## Implementation"),
    code("""import torch.nn.functional as F
from inference_lab import flash
torch.manual_seed(0)
q = torch.randn(1, 8, 1, 128)                      # one decode query, 8 heads
k, v = torch.randn(1, 8, 8192, 128), torch.randn(1, 8, 8192, 128)
ref = F.scaled_dot_product_attention(q, k, v)
for s in (1, 2, 8, 64):
    err = float((flash.split_kv_attention(q, k, v, s) - ref).abs().max())
    print(f"splits={s:3d}  max |split - SDPA| = {err:.2e}")"""),
    md("## Experiment: cascade attention for a shared prefix"),
    code("""prefix_k, prefix_v = k[:, :, :6000], v[:, :, :6000]
Q = torch.randn(4, 8, 1, 128)                      # 4 users, same prefix
o_pre, lse_pre = flash.partial_state(Q, prefix_k.expand(4, -1, -1, -1), prefix_v.expand(4, -1, -1, -1))
for i in range(4):
    ks, vs = torch.randn(1, 8, 300, 128), torch.randn(1, 8, 300, 128)
    o_suf, lse_suf = flash.partial_state(Q[i:i+1], ks, vs)
    o, _ = flash.merge_states(o_pre[i:i+1], lse_pre[i:i+1], o_suf, lse_suf)
    full = F.scaled_dot_product_attention(Q[i:i+1], torch.cat([prefix_k, ks], 2), torch.cat([prefix_v, vs], 2))
    print(f"user {i}: max |cascade - full| = {float((o - full).abs().max()):.1e}")"""),
    md("""## Interpretation
Any split is exact. Cascade attention reads the shared prefix once per batch instead of once per
request, which is the bandwidth saving FlashInfer's cascade kernels exploit.

## Limitations
PyTorch loops; no parallel hardware scheduling.

## Conclusion
One identity, the LSE merge, underlies split-KV decoding, cascade attention and context parallelism."""),
]

NOTEBOOKS["sota_04_kv_quant_kivi_turboquant"] = header(
    "S04 · KV quantization: per-token vs KIVI vs TurboQuant", "path step 11",
    "TurboQuant (arXiv:2504.19874; most-starred implementation github.com/0xSero/turboquant, GPL-3.0, linked only); "
    "KIVI (arXiv:2402.02750, github.com/jy-yuan/KIVI)") + [
    md("""## Objective
Compare three ways to quantize keys that have outlier channels: naive per-token groups, KIVI's
per-channel key groups, and TurboQuant's random rotation plus a fixed Gaussian codebook."""),
    md(r"""## Theory and equations
Group quantization error $\le \Delta/2$ with $\Delta = (\max-\min)/(2^b-1)$, so an outlier in a group hurts every element.
TurboQuant: $y = \sqrt{d}\,R\,x/\|x\|$ has nearly i.i.d. $\mathcal N(0,1)$ coordinates for a Haar-random $R$;
quantize each with the Lloyd-Max codebook; store $\|x\|$. Expected relative error $\approx \sqrt{D_b}$
($D_1 = 0.363$, $D_2 = 0.118$, $D_3 = 0.0345$, $D_4 = 0.0095$)."""),
    md("""## Diagram
```
per-token:    [ outlier ch | normal ch ... ]  one range per token  -> range blown up by the outlier
KIVI:         per channel across tokens         -> the outlier channel gets its own range
TurboQuant:   rotate -> outlier energy spread over all coords -> fixed N(0,1) codebook
```"""),
    md("## Implementation"),
    code("""from inference_lab import quant
k, v = quant.synthetic_kv()                          # 512 tokens, d=128, 4 outlier channels
print("Lloyd-Max N(0,1) 2-bit codebook:", quant.lloyd_max_gaussian(2))"""),
    md("## Experiment"),
    code("""q = torch.randn(16, 128, generator=torch.Generator().manual_seed(9))
full = quant.attention_out(q, k, v)
e = quant.rel_error
print(f"{'bits':>4} {'K tok':>7} {'K KIVI':>7} {'K TQ':>7} | {'out tok':>8} {'out KIVI':>9} {'out TQ':>7}")
for b in (4, 3, 2):
    kt, kc, ktq = quant.quantize_grouped(k, b, "per_token"), quant.quantize_grouped(k, b, "per_channel"), quant.turboquant(k, b)
    vt, vtq = quant.quantize_grouped(v, b, "per_token"), quant.turboquant(v, b)
    kk, vv = quant.kivi(k, v, b)
    print(f"{b:4d} {e(kt,k):7.3f} {e(kc,k):7.3f} {e(ktq,k):7.3f} | {e(quant.attention_out(q,kt,vt),full):8.3f} "
          f"{e(quant.attention_out(q,kk,vv),full):9.3f} {e(quant.attention_out(q,ktq,vtq),full):7.3f}")
print("bytes/elem at 2 bits: group-32 =", quant.bytes_per_elem(2, 32), " TurboQuant d=128 =", quant.turboquant_bytes_per_elem(2, 128))"""),
    md("## Results: sensitivity to outlier strength"),
    code("""scales = [0, 2, 5, 10, 15, 25]
rows = {"per-token": [], "KIVI per-channel": [], "TurboQuant": []}
for s_ in scales:
    kx, _ = quant.synthetic_kv(outlier_scale=s_)
    rows["per-token"].append(quant.rel_error(quant.quantize_grouped(kx, 3, "per_token"), kx))
    rows["KIVI per-channel"].append(quant.rel_error(quant.quantize_grouped(kx, 3, "per_channel"), kx))
    rows["TurboQuant"].append(quant.rel_error(quant.turboquant(kx, 3), kx))
for (name, ys), col in zip(rows.items(), [C["waste"], C["memory"], C["compute"]]):
    plt.plot(scales, ys, "o-", color=col, label=name)
plt.xlabel("outlier channel offset (sigma)"); plt.ylabel("3-bit key relative error"); plt.legend(); plt.show()"""),
    md("""## Interpretation
Per-token error grows with outlier strength. KIVI isolates outliers by grouping per channel.
TurboQuant removes the dependence differently: after rotation every vector looks Gaussian, so
one fixed codebook is near-optimal and its error stays near the Lloyd-Max value. The rotation
must be drawn independently of the data; a correlated "random" matrix breaks the Gaussian assumption.

## Limitations
Synthetic data; the paper's QJL stage for unbiased inner products is not implemented.

## Conclusion
Outliers decide KV quantization quality. Isolate them (KIVI) or spread them (TurboQuant)."""),
]

NOTEBOOKS["sota_05_kv_eviction"] = header(
    "S05 · KV eviction: StreamingLLM, H2O, SnapKV", "path step 11",
    "github.com/mit-han-lab/streaming-llm (7.3k stars, MIT); arXiv:2306.14048; arXiv:2404.14469; NVIDIA/kvpress") + [
    md("""## Objective
Compare three eviction rules at equal token budgets on a context with a few important tokens."""),
    md(r"""## Theory and equations
Error: $\|\operatorname{attn}(q, K_S, V_S) - \operatorname{attn}(q, K, V)\| / \|\operatorname{attn}(q,K,V)\|$ for the kept set $S$."""),
    md("""## Diagram
```
StreamingLLM: [sinks]..........................................[recent window]
H2O:          [heavy hitters by accumulated attention]          [recent window]
SnapKV:       [top tokens voted by the last w prompt queries, pooled]  [obs window]
```"""),
    md("## Implementation"),
    code("""from inference_lab import eviction as E
k, v, d, pos = E.needle_context()
q_obs, q_future = E.queries_toward(d, 32, seed=5), E.queries_toward(d, 16, seed=6)
hist = E.attention_probs(q_obs, k)
print("needle positions:", pos.tolist())"""),
    md("## Experiment"),
    code("""budgets = [64, 128, 256, 512]
res = {"StreamingLLM": [], "H2O": [], "SnapKV": []}
for B in budgets:
    res["StreamingLLM"].append(E.output_error(q_future, k, v, E.streaming(1024, B)))
    res["H2O"].append(E.output_error(q_future, k, v, E.h2o(hist, B, B // 2)))
    res["SnapKV"].append(E.output_error(q_future, k, v, E.snapkv(q_obs, k, B)))
for (name, ys), col in zip(res.items(), [C["waste"], C["memory"], C["compute"]]):
    plt.plot(budgets, ys, "o-", color=col, label=name)
    print(name, [round(y, 3) for y in ys])
plt.xscale("log", base=2); plt.yscale("log"); plt.xlabel("kept tokens (of 1024)"); plt.ylabel("relative output error")
plt.legend(); plt.show()"""),
    md("## Results: when attention is diffuse"),
    code("""k2, v2, d2, _ = E.needle_context(strength=4.0)
q2 = E.queries_toward(d2, 16, seed=6)
kept = (E.streaming(1024, 256), E.snapkv(E.queries_toward(d2, 32, seed=5), k2, 256))
print("diffuse, budget 256:", [round(E.output_error(q2, k2, v2, s), 3) for s in kept])"""),
    md("""## Interpretation
Content-aware rules find the needles; the content-blind window cannot. When attention is
diffuse no small budget works, which is why eviction quality is task-dependent.

## Limitations
One head, synthetic keys, one query distribution.

## Conclusion
Eviction is cheap and lossy; its quality depends on how peaked attention is for the task."""),
]

NOTEBOOKS["sota_06_speculative_decoding"] = header(
    "S06 · Speculative decoding: exactness, payoff and confidence-scheduled verification", "path step 12",
    "Leviathan et al. (arXiv:2211.17192); github.com/deepseek-ai/DeepSpec (7.2k stars, MIT: EAGLE-3, DFlash, DSpark); "
    "DSpark (arXiv:2607.05147)") + [
    md("""## Objective
Check that speculative sampling preserves the target distribution, map the speedup surface, and
model the newest idea: when drafters propose whole blocks, verify only the confident part."""),
    md(r"""## Theory and equations
Accept $x\sim q$ with $\min(1, p/q)$; else sample $\operatorname{norm}(\max(0,p-q))$.
$\mathbb{E}[\text{tokens}] = (1-\alpha^{\gamma+1})/(1-\alpha)$; speedup $= \mathbb{E}/(\gamma c + 1)$.
With position-dependent acceptance, verifying draft $j$ is worth $\prod_{i\le j}\alpha_i$ tokens; under a shared
slot budget, verify the drafts with the largest cumulative confidence."""),
    md("""## Diagram
```
block drafter (DFlash / DSpark):  t1 t2 t3 t4 t5 t6 t7  in one draft pass; confidence decays ->
verify:  request A (confident): t1..t6      request B (unsure): t1..t2      same total slots
```"""),
    md("## Implementation"),
    code("""from inference_lab import specdec as SD
gen = torch.Generator().manual_seed(0)
p = torch.tensor([0.45, 0.25, 0.15, 0.10, 0.05]); q = torch.full((5,), 0.2)
counts = torch.zeros(5); acc = 0; n = 50_000
for _ in range(n):
    x, a = SD.speculative_step(p, q, gen); counts[x] += 1; acc += a
print("target p    :", p.tolist()); print("emitted (MC):", (counts / n).round(decimals=4).tolist())
print("acceptance MC", acc / n, " theory 1-TV", SD.acceptance_rate(p, q))"""),
    md("## Experiment 1: expected tokens, formula vs Monte Carlo"),
    code("""for a in (0.5, 0.7, 0.9):
    print(a, [(g, round(SD.expected_tokens(a, g), 3), round(SD.simulate_chain(a, g, 100_000), 3)) for g in (2, 4, 8)])"""),
    md("## Experiment 2: confidence-scheduled vs fixed-length verification (*theoretical*)"),
    code("""gen = torch.Generator().manual_seed(3)
N, gamma = 64, 8
base = torch.rand(N, 1, generator=gen) * 0.45 + 0.5                # per-request draft quality
alphas = (base * torch.linspace(1.0, 0.8, gamma)).clamp(max=0.99)    # acceptance decays along the block
for slots_per_req in (2, 3, 4, 6, 9):
    budget = N * slots_per_req
    lengths, tot = SD.confidence_schedule(alphas, budget)
    k, tot_fixed = SD.fixed_schedule(alphas, budget)
    print(f"{slots_per_req} slots/request: fixed k={k} -> {tot_fixed:6.1f} tokens; "
          f"confidence-scheduled -> {tot:6.1f} tokens (lengths {int(lengths.min())}..{int(lengths.max())})")"""),
    md("""## Interpretation
The emitted distribution matches $p$. Under a tight slot budget, giving long verifications to
confident requests and short ones to unsure requests yields more tokens per batch than a
uniform draft length. That is the scheduling problem DSpark addresses for high-concurrency serving.

## Limitations
I.i.d. or fixed per-position acceptance; real acceptance depends on content.

## Conclusion
Speculation never changes outputs. At high concurrency, what you verify matters as much as what you draft."""),
]

NOTEBOOKS["sota_07_chunked_prefill"] = header(
    "S07 · Chunked prefill: choosing the token budget", "path step 9",
    "Sarathi-Serve (arXiv:2403.02310); vLLM V1 scheduler") + [
    md("""## Objective
Sweep the chunked-prefill token budget and watch TTFT and p99 ITL trade off."""),
    md(r"""## Theory and equations
Each iteration processes at most $\tau$ tokens. A $T_p$-token prompt needs
$\lceil T_p/(\tau - B_{\text{dec}})\rceil$ iterations; iteration time grows with $\tau$."""),
    md("""## Diagram
```
tau small: many short iterations -> smooth ITL, slower prefill (higher TTFT)
tau large: few long iterations   -> faster prefill, longer gaps between decode tokens
```"""),
    md("## Implementation"),
    code("""from inference_lab import scheduler as S
trace = S.poisson_trace(150, rate=15.0, seed=2, prompt=(512, 4096))"""),
    md("## Experiment (*theoretical*, roofline-timed)"),
    code("""budgets = [128, 256, 512, 1024, 2048, 4096]
rs = [S.simulate(trace, "chunked", token_budget=b) for b in budgets]
for b, r in zip(budgets, rs):
    print(f"tau={b:5d}  TTFT mean {r.ttft_mean:6.3f}s  ITL p99 {1e3*r.itl_p99:6.1f} ms  throughput {r.throughput_tok_s:6.0f} tok/s")
fig, ax = plt.subplots(); ax2 = ax.twinx()
ax.plot(budgets, [r.ttft_mean for r in rs], "o-", color=C["compute"]); ax.set_ylabel("mean TTFT (s)", color=C["compute"])
ax2.plot(budgets, [1e3 * r.itl_p99 for r in rs], "s-", color=C["memory"]); ax2.set_ylabel("p99 ITL (ms)", color=C["memory"])
ax.set_xscale("log", base=2); ax.set_xlabel("token budget per iteration"); plt.show()"""),
    md("## Results: separate the two effects (unlimited KV memory)"),
    code("""rs2 = [S.simulate(trace, "chunked", token_budget=b, num_blocks=10**6) for b in budgets]
for b, r in zip(budgets, rs2):
    print(f"tau={b:5d}  TTFT mean {r.ttft_mean:6.3f}s  ITL p99 {1e3*r.itl_p99:6.1f} ms  preemptions {r.preemptions}")"""),
    md("""## Interpretation
TTFT is U-shaped in the budget. A tiny budget makes every iteration pay a full read of the
weights for a handful of prompt tokens, so prefill crawls and the queue grows. A huge budget makes
every iteration long, so ITL rises in proportion and each prompt waits behind other prompts'
chunks. With a realistic KV pool, memory pressure and preemptions add to TTFT on top of that;
the unlimited-memory run isolates the scheduling effect.

## Limitations
Roofline timing; no fixed per-iteration CPU overhead, which would penalise small budgets more.

## Conclusion
Choose the budget from the ITL target first (iteration time grows with it), then check TTFT;
if TTFT is dominated by memory pressure, the fix is more KV capacity, not a different budget."""),
]

NOTEBOOKS["sota_08_pd_disaggregation"] = header(
    "S08 · Prefill/decode disaggregation: is the KV transfer worth it?", "path step 12",
    "DistServe (arXiv:2401.09670); Mooncake (arXiv:2407.00079, github.com/kvcache-ai/Mooncake)") + [
    md("""## Objective
Quantify the KV transfer that disaggregation adds, relative to the prefill it follows."""),
    md(r"""## Theory and equations
$t_{\text{xfer}} = T_p m_{\text{tok}}/\beta_{\text{link}}$; ratio $t_{\text{xfer}}/t_{\text{prefill}} \propto m_{\text{tok}}/P$ (prompt length cancels when prefill is compute-bound)."""),
    md("""## Diagram
```
[prefill pool] --KV (T_p x m_tok bytes) over RDMA--> [decode pool] --> tokens
   sets TTFT                                           sets ITL
```"""),
    md("## Implementation"),
    code("""from inference_lab import disagg as D, kv_math as K
models = [K.LLAMA3_8B, K.LLAMA3_70B, K.DEEPSEEK_V3]"""),
    md("## Experiment (*theoretical*)"),
    code("""for m in models:
    print(m.name, f"prefill 8K on H100 roofline: {1e3*D.prefill_time(m, K.H100_SXM, 8192):.0f} ms")
    for link, bw in D.LINKS_GBS.items():
        print(f"   {link:22s} transfer {1e3*D.transfer_time(m, 8192, bw):7.1f} ms   ratio {D.transfer_ratio(m, K.H100_SXM, 8192, bw):.3f}")"""),
    md("## Results: ratio vs prompt length"),
    code("""Ts = [256, 1024, 4096, 16384, 65536]
for m, col in zip(models, [C["waste"], C["memory"], C["compute"]]):
    plt.plot(Ts, [D.transfer_ratio(m, K.H100_SXM, T, 50.0) for T in Ts], "o-", color=col, label=m.name)
plt.xscale("log", base=2); plt.yscale("log"); plt.axhline(0.1, color=C["ink"], ls="--", lw=1)
plt.xlabel("prompt tokens"); plt.ylabel("transfer / prefill (IB NDR 400G)"); plt.legend(); plt.show()"""),
    md("""## Interpretation
Big models with compact caches (GQA-8 70B, MLA) make transfer a few percent of prefill on RDMA.
Small models on slow links do not. Short prompts are memory-bound in prefill, which raises the
ratio again.

## Limitations
Roofline prefill (an optimistic lower bound on prefill time makes ratios pessimistic); no
layer-wise overlap, which real systems use to hide transfer.

## Conclusion
Disaggregate large models on fast fabrics; keep small models co-located with chunked prefill."""),
]


def main(args: list[str]) -> None:
    execute = "--execute" in args
    filters = [a for a in args if a != "--execute"]
    OUT.mkdir(exist_ok=True)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3") if execute else None
    for name, cells in NOTEBOOKS.items():
        if filters and not any(f in name for f in filters):
            continue
        nb = nbformat.v4.new_notebook(cells=cells, metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"}})
        if ep is not None:
            ep.preprocess(nb, {"metadata": {"path": str(OUT)}})
        nbformat.write(nb, OUT / f"{name}.ipynb")
        print("executed" if ep else "wrote", name)


if __name__ == "__main__":
    main(sys.argv[1:])
