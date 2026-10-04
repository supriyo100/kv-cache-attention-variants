# KV Cache, Attention Variants & SOTA Routes — Deep-Dive Explanation

> Companion to the course repo [sourangshupal/kv-cache-attention-variants](https://github.com/sourangshupal/kv-cache-attention-variants).
> Part A explains the theory from tokens to the KV cache. Part B walks the repository module by module.
> Part C covers today's fine-tuning and inference stacks, including methods the repo doesn't cover.
> The visual companion is `docs/explain.ecalidraw` (open it with the Excalidraw extension).

**The one-line idea:** *the KV cache spends GPU memory to save compute.* Every attention variant in this
repo (MQA, GQA, MLA) is a way to make that memory bill smaller. Every serving trick (FlashAttention,
PagedAttention, prefix caching) is a way to move or manage that memory faster.

---

## Part A — From tokens to the KV cache

### A1. Tokens and embeddings

An LLM never sees text. It sees **token ids** that a tokenizer (BPE, SentencePiece, or tiktoken) produces:

```
"The patient was diagnosed with diabetes"
 → [The] [ patient] [ was] [ diagnosed] [ with] [ diabetes]
 → [791, 8893, 574, 29704, 449, 20335]          (ids are illustrative)
```

An embedding table `E ∈ R^{vocab × d_model}` maps each id to a vector. So `x = E[ids]` has shape
`(batch, seq, d_model)`. In the repo this is `nn.Embedding(vocab, d_model)`; see
`naive_decode.py` and `lessons/session01_naive_decoding.py`.

**Special tokens** are reserved ids with structural meaning:

| Token | Role |
|---|---|
| `<BOS>` / `<s>` | Begin-of-sequence. Gives the first real token something to attend to, and models often use it as an "attention sink". |
| `<EOS>` / `</s>` / `<\|eot_id\|>` | End-of-sequence. Generation stops when the model samples it. |
| `<PAD>` | Fills shorter sequences in a batch. It is masked out and never attended to. |
| Chat-template tokens (`<\|im_start\|>`, `<\|start_header_id\|>` …) | Mark role boundaries (system/user/assistant). Fine-tuning has to keep these consistent with inference. |

### A2. Self-attention

Each token projects itself into three roles:

```
Q = x·W_q   "what am I looking for?"
K = x·W_k   "what do I contain?"     (the label)
V = x·W_v   "what do I hand over?"   (the payload)
```

### A3. Scaled dot-product attention (SDPA)

$$
\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

* `QKᵀ` gives a `(seq × seq)` matrix of similarity scores.
* `/√d_k` keeps the variance of those scores near 1. Without it, the softmax saturates into one-hot outputs and gradients vanish.
* `M` is the mask (see A4).
* The softmax turns each row into a probability distribution, and multiplying by `V` gives a weighted mix of payloads.

**Multi-head:** the model dimension is split into `h` heads of `head_dim = d_model / h`, and each head
attends independently. The repo keeps every tensor as `(batch, heads, seq, head_dim)`, which is the
layout `F.scaled_dot_product_attention` expects (`attention/mha.py` `_split_heads`).

### A4. Causal masking and padding

A decoder may not look at the future. The mask is an upper-triangular block of `-inf`:

```python
T = 5
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)   # True above the diagonal
scores = scores.masked_fill(mask, float("-inf"))                    # softmax(-inf) = 0
```

```
        k0  k1  k2  k3  k4
  q0  [  ✓   ✗   ✗   ✗   ✗ ]
  q1  [  ✓   ✓   ✗   ✗   ✗ ]
  q2  [  ✓   ✓   ✓   ✗   ✗ ]
  q3  [  ✓   ✓   ✓   ✓   ✗ ]
  q4  [  ✓   ✓   ✓   ✓   ✓ ]
```

`F.scaled_dot_product_attention(..., is_causal=True)` builds this mask internally (or fuses it, as
the Flash kernels do). The repo uses this everywhere.

**Left padding for batched generation.** For decoder-only models you pad on the **left**:

```
[PAD][PAD][The][cat]      ← the next token is predicted from position -1, which is real
[Hi ][there][my][dog]
```

If you pad on the right, position `-1` is a PAD for short rows, so the model would predict from
garbage. Left padding also needs an `attention_mask` that zeros the PADs and `position_ids`
computed with `attention_mask.cumsum(-1) - 1`, so RoPE positions start at 0 on the first real token.
(For training, right padding is fine because loss is computed on every position.)

**Subtle SDPA detail the repo relies on:** when `q_len ≠ k_len`, PyTorch's `is_causal=True` aligns the
mask to the *top-left* corner. On a decode step (`q_len = 1`, `k_len = past+1`) that would hide
almost everything. So the repo sets `is_causal = past_kv is None`: causal on the prompt, no mask on
single-token decode (one query may see every key, since all keys are in the past).

### A5. Autoregressive generation

$$
p(x_1,\dots,x_T)=\prod_{t} p(x_t \mid x_{<t})
$$

The model is trained to predict the next token given everything before it. At inference that
gives a strictly **sequential loop**:

```
ids = prompt
repeat:
    logits = model(ids)[:, -1, :]        # only the LAST position matters
    next   = sample(logits)              # greedy / top-k / top-p / temperature
    ids    = ids + [next]
until next == <EOS>  or  len(new) == max_new_tokens
```

Key properties:

* **One token per forward pass.** Token `t+1` depends on token `t`, so decode cannot be parallelised
  across time. (Speculative decoding gets around this, see C2.)
* **Stopping:** at `<EOS>`, at a stop string, or at `max_new_tokens`. The model's context window
  (prompt + generated tokens ≤ `max_position_embeddings`) is the hard limit.
* **Sampling** (the repo uses greedy `argmax`):

  | Strategy | What it does |
  |---|---|
  | Greedy | `argmax`. Deterministic, but tends to loop. |
  | Temperature `τ` | `softmax(logits/τ)`. `τ<1` sharpens, `τ>1` flattens. |
  | **Top-k** | Keep the `k` highest logits, renormalise, then sample. |
  | Top-p (nucleus) | Keep the smallest set whose cumulative probability is ≥ `p`. |
  | Min-p | Keep tokens with `p ≥ min_p · p_max`. Adapts to how confident the model is. |
  | Repetition / frequency penalties | Down-weight tokens already generated. |

### A6. Prefill vs decode, the two phases of inference

| | **Prefill** | **Decode** |
|---|---|---|
| Input | The whole prompt (N tokens) | 1 new token |
| Parallelism | All N tokens at once (one big causal matmul) | Sequential, one step per token |
| Bound by | **Compute** (FLOPs, tensor cores) | **Memory bandwidth** (reading weights and the KV cache) |
| Metric | TTFT, time-to-first-token | TPOT / ITL, time-per-output-token |
| Writes KV cache? | Yes, N entries per layer | Yes, 1 entry per layer per step |

**Why a token takes ~20–30 ms.** At batch 1, each decode step has to stream *all the weights*
plus *the KV cache* out of HBM:

$$
t_\text{token} \gtrsim \frac{\text{weight bytes} + \text{KV bytes}}{\text{HBM bandwidth}}
$$

A 7B fp16 model is ~14 GB. On an H100 (~3.35 TB/s) that is about 4 ms as a lower bound. A 70B model
sharded across GPUs, plus kernel launches, communication and sampling overhead, lands in the familiar
20–30 ms/token range. Doing less arithmetic doesn't help much. Moving fewer bytes does: quantise weights and KV,
shrink the KV cache (GQA/MLA), batch more requests per weight read, or verify several tokens per
weight read (speculative decoding).

### A7. The problem: naive decoding recomputes the past

Without a cache, step `t` re-embeds and re-projects K/V for all `t` previous tokens:

```
step 1: [The patient has diabetes and]            → compute K,V for 5 → "requires"
step 2: [The patient has diabetes and requires]   → compute K,V for 6 → "insulin"
step 3: [... requires insulin]                    → compute K,V for 7 → "therapy"
```

Total work ≈ `Σ t` = O(T²) projections. Past K and V never change (causal attention means an
old token can't see new tokens), so recomputing them is pure waste.

### A8. The fix: the KV cache

Store each layer's K and V once, append one row per step, and compute only the **new** token's Q/K/V:

```
step t:   q_t, k_t, v_t  = project(x_t)            # 1 token
          K_cache ← concat(K_cache, k_t)           # (b, h_kv, t, d)
          V_cache ← concat(V_cache, v_t)
          out_t   = softmax(q_t · K_cacheᵀ / √d) · V_cache
```

* **Why K and V but not Q?** Q belongs to the *current* token: it asks "what do I need now?" and is
  used once. K and V of past tokens are reused at every future step.
* Per-step attention becomes O(t) (1 query × t keys) instead of recomputing the whole O(t²) history.
* The price is **memory**, and it grows linearly with context length and batch size.

### A9. KV cache memory math (`memory_calc.py`)

$$
\boxed{\text{KV bytes} = 2 \times L \times H_{kv} \times d_{head} \times \text{bytes} \times T \times B}
$$

The `2` counts K and V, `L` is layers, `H_kv` is the number of **KV** heads (not query heads), `T` is
sequence length, and `B` is batch.

| Model shape | Per token | 8K ctx, batch 1 | 8K ctx, batch 32 |
|---|---|---|---|
| Llama-2-7B, MHA (32L, 32 KV heads, 128) | 512 KiB | 4 GiB | 128 GiB |
| Llama-3-8B, GQA-8 (32L, 8 KV heads) | 128 KiB | 1 GiB | 32 GiB |
| 70B, MHA-64 (session02 config) | 2.5 MiB | 20 GiB | 640 GiB |
| Llama-3-70B, GQA-8 (80L, 8 KV heads) | 320 KiB | **2.5 GiB (2.68 GB)** | **80 GiB (85.9 GB)** |
| DeepSeek-V2, MLA (60L, 512 latent + 64 RoPE) | ~67.5 KiB | ~0.53 GiB | ~17 GiB |

(Numbers are fp16 and were checked against `kv_cache_bytes`.) The lesson: **at long context or high
concurrency, the KV cache, not the weights, decides how many users fit on a GPU.** GPU memory ≈
weights + KV cache + activations/workspace + runtime overhead.

---

## Part B — The repository, in depth

### B1. Design philosophy

* It's a **10-session course**. Each optimisation exists because the previous approach fails in some way,
  and the dependency chain is drawn in `docs/diagrams/D01_curriculum_dependency.html`.
* **One source of truth:** the math lives in `src/kv_cache_variants/`. `lessons/` is purely
  presentation (rich-terminal tables) and imports the real modules without re-implementing them.
* **Mac-first / CPU-first:** `sdpa_backends.default_device()` picks MPS or CPU. The CUDA-only topics
  (real FlashAttention kernels, vLLM) are taught with CPU stand-ins. The README mentions Colab notebooks in `colab/`, but
  **that folder isn't in the repo**.
* Tooling: `uv` + `pyproject.toml` (torch ≥ 2.13, rich, jupyter), MkDocs-Material docs auto-deployed by
  `.github/workflows/docs.yml`, and 12 self-contained HTML diagrams in `docs/diagrams/`.

### B2. Module map

```
src/kv_cache_variants/
├── naive_decode.py      S1  TinyCausalAttention + generate_naive (no cache)      ─┐
├── memory_calc.py       S2  kv_cache_bytes(), human_bytes()                        │ used by S2, S6, S10
├── attention/
│   ├── mha.py           S3  MultiHeadAttention(x, past_kv) → (out, (K,V))           │ reference cache layout
│   ├── gqa.py           S5  GroupedQueryAttention(num_kv_groups)                    │ generalises MHA & MQA
│   ├── mqa.py           S4  MultiQueryAttention = GQA(num_kv_groups=1)             │ thin wrapper
│   └── mla.py           S6  MultiHeadLatentAttention (latent + decoupled RoPE)    ─┤ imports rope.apply_rotary
├── rope.py              S7  build_rope_cache, apply_rotary(offset), PI / NTK / YaRN│
├── sdpa_backends.py     S8–9 run_sdpa(backend), available_backends(), default_device
├── bench.py             S8–10 benchmark(fn, warmup, iters) with device sync
└── lessons/session01…10_*.py   rich-terminal walkthroughs (presentation only)
```

### B3. Session by session

**S0 — Foundations** (`docs/00_foundations.md`, notebook 00): tokens, embeddings, SDPA by hand on a
3×4 example, causal masking, the autoregressive loop, and the decoder block (attention → residual →
norm → MLP).

**S1 — Naive decoding** (`naive_decode.py`). `generate_naive` calls `embed(ids)` and `model(x)` on the
**whole** sequence every step and keeps only `hidden[:, -1, :]`. The lesson wraps `attn.forward` with a counter
to show the tokens recomputed per step (4, 5, 6, … 11 → 60 total vs 8 with a cache, so 7.5× more
work). That gap grows quadratically with length.

**S2 — Memory math** (`memory_calc.py`). `per_token = 2·H_kv·d·L·bytes`. The lesson tabulates MHA vs
GQA-8 vs MQA at 2K/8K/32K for 7B/13B/70B shapes. The takeaway is that cache size is **linear in `H_kv`**, which motivates
everything after it.

**S3 — MHA with incremental cache** (`attention/mha.py`). This is the reference implementation:

```python
q, k, v = split(q_proj(x)), split(k_proj(x)), split(v_proj(x))   # (b, h, s, d)
if past_kv: k = cat([past_k, k], dim=2); v = cat([past_v, v], dim=2)
out = sdpa(q, k, v, is_causal = past_kv is None)
return out_proj(merge(out)), (k, v)            # caller passes (k, v) back next step
```

The cache is a tuple of tensors `(b, num_heads, seq, head_dim)`, and `KVCache = list[tuple]` is the
per-layer list. `torch.cat` on every step is O(t) copying, which is fine for teaching. Production engines
preallocate a buffer or use pages (S10).

**S4 — MQA** (`attention/mqa.py`). One shared K/V head for all query heads, so the cache shrinks by `h×`.
It is implemented as `GroupedQueryAttention(num_kv_groups=1)`, which makes MQA a *special case* of GQA.
The cost is quality, because all heads have to read the same "label/payload" space. PaLM and Falcon used it.

**S5 — GQA** (`attention/gqa.py`). `num_kv_groups` is a dial: `= num_heads` gives MHA, `= 1` gives MQA.

```python
k = split(k_proj(x), num_kv_groups)          # project to FEWER heads
cached_kv = (k, v)                           # ← only num_kv_groups heads are stored
k = k.repeat_interleave(group_size, dim=1)   # expand at attention time only
```

The key distinction is **storage vs compute.** The cache holds `G` heads, and the expansion happens only
for the math. (Production kernels avoid even this materialisation, and PyTorch SDPA has `enable_gqa=True`.)
Llama-2-70B, Llama-3, Mistral and Qwen all use GQA (usually 8 KV heads). The chart at
`docs/05_gqa_files/05_gqa_2_1.png` plots this spectrum.

**S6 — MLA, DeepSeek-V2** (`attention/mla.py`). This is a different axis: instead of fewer heads, it
caches a **low-rank latent per token**.

```
x ─► kv_down_proj ─► c_t (latent_dim)  ──CACHE──► k_up_proj ─► K_content (h × d)
                                                   v_up_proj ─► V          (h × d)
x ─► k_rope_proj ─► k_rope (rope_dim, shared by all heads) ──CACHE──► RoPE ─► concat to K
x ─► q_proj + q_rope_proj ─► Q = [q_content ‖ RoPE(q_rope)]
```

* The cache is `(latent, k_rope)`, about `latent_dim + rope_dim` values per token per layer, **independent of
  the head count**. All heads still get full-rank K/V after up-projection, which is why MLA can match or beat
  MHA quality with a cache smaller than GQA's.
* **Decoupled RoPE:** a rotation depends on position, so it can't be folded into the static `W_up`
  matrices. A small separate RoPE key is cached alongside the latent.
* In production, DeepSeek "absorbs" `W_uk` into `W_q` and `W_uv` into `W_o`, so attention runs directly
  on the latent and K/V are never materialised. The repo up-projects the full latent at every step for
  clarity.

**S7 — RoPE** (`rope.py`). Position is encoded by **rotating** pairs of dimensions of Q and K by angle
`θ_i·pos`, with `θ_i = base^{-2i/d}`. The dot product `q_m·k_n` then depends only on `m − n`, so it is
relative. Two properties matter for caching:

* RoPE is applied **before** K enters the cache, so cached keys already carry their position.
* At decode, the new token is rotated at its true position: `apply_rotary(x, cos, sin, offset=past_len)`.
  The `_demo` asserts that this matches rotating the full sequence.

Context extension, i.e. running past the trained length:

| Method | Function | Idea |
|---|---|---|
| Position Interpolation | `linear_scaled_rope_cache` | Divide positions by `s`. Simple, but blurs local detail. |
| (Dynamic) NTK-aware | `dynamic_ntk_rope_cache` | Raise the base, which stretches low frequencies more than high ones. |
| YaRN | `yarn_rope_cache` | Ramp per frequency: keep high-frequency (local) bands and interpolate low-frequency (global) bands. Real YaRN also scales attention temperature (`0.1·ln s + 1`), which this simplified version omits. |

`docs/07_rope_files/07_rope_7_0.png` visualises the rotation and frequency bands.

**S8 — FlashAttention** (`lessons/session08_flashattention.py` + `sdpa_backends.py` + `bench.py`).
FlashAttention's speedup comes from **IO**, not from fewer FLOPs. It tiles Q/K/V into SRAM blocks and uses
an **online softmax** (a running max and running sum, rescaled per tile), so the `N×N` score matrix is
never written to HBM. Memory drops from O(N²) to O(N) and the result is still *exact* attention. The repo
can't run the CUDA kernel locally, so it benchmarks SDPA backends instead (diagram D10 shows the tiling).

**S9 — PyTorch SDPA migration** (`sdpa_backends.py`). `F.scaled_dot_product_attention` is one API with
several backends: `MATH` (reference, any device), `FLASH_ATTENTION`, `EFFICIENT_ATTENTION` (xFormers-style
mem-efficient), and `CUDNN_ATTENTION`. `run_sdpa` *forces* a backend through `sdpa_kernel(...)`, and when the
device, dtype or shape is unsupported it logs the `RuntimeError` and falls back to math. That failure is the
lesson: backend choice depends on hardware and shape. Diagram D11 shows the dispatch.

**S10 — PagedAttention / vLLM** (`lessons/session10_pagedattention.py`). This is a CPU **allocation**
simulation. 32 requests with random real lengths (50–2048) are compared under:

* *Naive:* preallocate `max_len` contiguous KV per request. Waste = `max_len − actual`, plus fragmentation.
* *Paged:* allocate fixed `block_size = 16`-token blocks on demand through a per-sequence **block table**
  (logical block → physical block), the way OS virtual memory works. Waste is at most 15 tokens per request.

This is what lets vLLM pack many more concurrent sequences per GPU and enables **continuous batching**
(sequences join and leave every iteration), **copy-on-write** prefix sharing, and parallel sampling and
beam search over shared blocks. Diagram D12 shows the block table.

### B4. Variant comparison (the whole repo in one table)

| Variant | What's cached per token per layer | Cache vs MHA | Quality | Used by |
|---|---|---|---|---|
| MHA | `2·h·d` | 1× | Best baseline | GPT-2/3, Llama-1, Llama-2-7B/13B |
| GQA-G | `2·G·d` | `G/h` (e.g. 8/64 = 1/8) | ≈ MHA | Llama-2-70B, Llama-3/4, Mistral, Qwen |
| MQA | `2·d` | `1/h` | Some loss | PaLM, Falcon, StarCoder |
| MLA | `d_c + d_rope` (e.g. 512 + 64) | ~1/57 of DeepSeek-V2's 128-head MHA | ≥ MHA (reported) | DeepSeek-V2/V3/R1, Kimi K2 |

`docs/diagrams/D05_variant_cache_comparison.html` and `D06_gqa_spectrum.html` show this visually.

### B5. Issues and caveats found while reading the code

1. **MLA's byte count is 2× too high in session 06.** `kv_cache_bytes(num_layers, 1, latent_dim + rope_dim, …)`
   multiplies by the K+V factor of 2, but MLA caches **one** latent plus one RoPE key, not separate K and V.
   The true value is `L·(d_c + d_rope)·bytes·T`. MLA still comes out smallest in the table, just by less than it should.
2. **`dynamic_ntk_rope_cache` line 1 is a no-op.** `base * ((s/t) − (s/t − 1))` equals `base * 1`. The second
   line, `base · (s/t)^{d/(d−2)}`, does the actual NTK scaling.
3. **`is_causal = past_kv is None`** is correct only when each decode step adds **one** token. Chunked prefill,
   or feeding several new tokens with a cache, would leave those new tokens mutually un-masked. Production code passes
   an explicit offset-aware mask.
4. **MLA re-rotates every cached `k_rope` at every step** (`offset=0` over the full sequence) because it caches
   un-rotated keys. The result is correct, just O(t) extra work per step. Caching post-rotation keys would avoid it.
5. **`torch.cat` cache growth** reallocates every step. That's fine for teaching, but it's exactly the
   fragmentation and copying problem S10 exists to solve.
6. The README points to `colab/*.ipynb` (real FlashAttention-2 and vLLM benchmarks), but that folder isn't in the repo.
   `teaching_notebooks/` covers sessions 00–04 only.

### B6. How to run it

```bash
uv sync --all-extras
uv run python -m kv_cache_variants.lessons.session01_naive_decoding   # … session10_pagedattention
uv run python -m kv_cache_variants.attention.mla                      # each module has a _demo()
uv run mkdocs serve                                                   # docs + diagrams at :8000
```

---

## Part C — SOTA routes

The two halves fit together like this: **fine-tuning decides *what* the model learns; inference
engineering decides *how cheaply* you can serve it.** Plan both together. For example, a LoRA you
intend to serve multi-tenant shouldn't be merged, and a model you will run in FP8 should be evaluated in FP8.

### C1. Fine-tuning routes

#### Decide first: do you need fine-tuning at all?

```
Need fresh / private facts?          → RAG first (fine-tuning is bad at injecting facts)
Model doesn't "speak" the domain?    → Continued pre-training (DAPT) on raw domain text
Need format / tone / tool-calling?   → SFT (LoRA usually enough)
Have chosen-vs-rejected pairs?       → Preference optimisation (DPO family)
Have a verifiable reward (tests, math, schema)? → RL (GRPO family, "RLVR")
Need a small fast model?             → Distillation from a big teacher
```

#### Route table

| # | Route | What it does | When to use | Tooling |
|---|---|---|---|---|
| 1 | **Full fine-tuning** | Updates every weight. Needs weights + grads + Adam states (~16 bytes/param in mixed precision). | Large high-quality datasets, big behaviour shifts, budget available. | FSDP2, DeepSpeed ZeRO-3, Megatron-LM, torchtune |
| 2 | **LoRA** | `W' = W + BA` with rank `r ≪ d`. W is frozen and only A and B train (<1% of params). Can be merged for zero-latency serving. | Default PEFT starting point. | HF PEFT, TRL, Unsloth, Axolotl, LLaMA-Factory |
| 3 | **QLoRA** | Base model in 4-bit **NF4** + double quantisation + paged optimisers, LoRA in bf16. A 70B model trains on ~48 GB. | Limited GPUs. The most practical route for most teams. | bitsandbytes + PEFT, Unsloth |
| 4 | **DoRA** | Splits W into *magnitude × direction* and applies LoRA to the direction. | Closes some of the LoRA-to-full-FT gap at the same rank. | PEFT `use_dora=True` |
| 5 | **PiSSA** | Initialises A and B from W's top singular vectors (the residual stays frozen). | Faster convergence, and less quantisation error than QLoRA in some settings. | PEFT `init_lora_weights="pissa"` |
| 6 | **LoftQ** | Jointly picks the quantised W and the LoRA init to minimise quantisation error. | 4-bit / 2-bit QLoRA where quality drops. | PEFT `LoftQConfig` |
| 7 | rsLoRA / LoRA+ | Rank-stabilised scaling `α/√r`, and a different learning rate for B vs A. | High ranks; faster or more stable training. | PEFT flags |
| 8 | **Continued pre-training (DAPT)** | Next-token loss on raw domain corpora (pharma, SAP, supply chain) **before** SFT. Mix in general data to avoid forgetting. | Domain vocabulary and style the model doesn't know. | Megatron, torchtune, Unsloth (CPT mode) |
| 9 | **SFT** | Supervised `(prompt → answer)` with loss on assistant tokens only and the correct chat template. | Format, tool calls, persona, task skills. | TRL `SFTTrainer` |
| 10 | **DPO** and family (IPO, KTO, ORPO, SimPO) | Learns from chosen/rejected pairs with no reward model or PPO loop. KTO needs only 👍/👎. ORPO and SimPO need no reference model. | You have preference data. Cheap alignment after SFT. | TRL `DPOTrainer`, `ORPOTrainer`, `KTOTrainer` |
| 11 | **GRPO / RLVR** (and DAPO, Dr. GRPO, GSPO variants) | Samples a group of answers per prompt, scores them with a **verifiable reward**, and uses group-relative advantages (no critic). This is how DeepSeek-R1-style reasoning is trained. | Math, code, SQL/Text2SQL, JSON-schema, tool use, agents. | TRL `GRPOTrainer`, verl, OpenRLHF, Unsloth |
| 12 | PPO-RLHF | Classic reward model + PPO. | Rarely the first choice now, since it's heavy and unstable. | OpenRLHF, verl |
| 13 | **Knowledge distillation** | A small student learns from a big teacher's outputs or logits (sequence-level or on-policy distillation). | Cheap, fast deployment models. | TRL `GKDTrainer`, custom |
| 14 | Model merging | Combines fine-tunes in weight space (TIES, DARE, SLERP). | Blending skills without retraining. | mergekit |

**Practical stack in 2026:** `Base → (DAPT) → SFT with LoRA/QLoRA → DPO or GRPO → eval → quantise
(FP8/INT4/NVFP4) → serve with vLLM/SGLang (multi-LoRA or merged)`. Target `all-linear` modules
(`q,k,v,o,gate,up,down`) for QLoRA-style quality. Evaluate in the **same precision and engine** you serve
with.

**Where fine-tuning meets the KV cache:**

* A LoRA on `k_proj`/`v_proj` changes K and V, so a prefix cache built with the base model **can't be
  reused** for that adapter. Engines key prefix caches by adapter for this reason.
* **Activated LoRA (aLoRA)** turns the adapter on only *after* an invocation token, so the base model's KV
  for the shared prefix stays valid and can be reused. That's useful for agent pipelines that switch between
  base, verifier and specialist adapters.
* Multi-LoRA serving (vLLM `--enable-lora`, S-LoRA/Punica kernels) keeps one base model in memory and
  many adapters per batch. One base plus N adapters costs far less than N full models.

### C2. Inference routes

Each technique below targets a specific bottleneck:

| Layer | Route | Bottleneck it attacks | Notes / SOTA |
|---|---|---|---|
| Architecture | **GQA / MQA / MLA** (this repo) | KV size | MLA is the strongest. Some teams convert GQA models to MLA after training (e.g. TransMLA, MHA2MLA). |
| Architecture | Sliding-window / hybrid local-global attention | KV grows with T | Mistral, Gemma 2/3 (interleaved local:global), gpt-oss |
| Architecture | Hybrid SSM / linear attention | O(T) KV entirely | Jamba, Mamba-2 hybrids, Qwen3-Next-style gated linear attention. Constant-size state. |
| Architecture | MoE + expert parallelism | FLOPs per token | DeepSeek-V3, Qwen3-MoE, Llama 4, gpt-oss |
| Kernels | **FlashAttention-2/3** (and FA-4 on Blackwell) | HBM traffic | FA-3 uses Hopper async/TMA + FP8. FlashInfer, FlashMLA and FlashDecoding (split-KV for decode). |
| Memory mgmt | **PagedAttention** | Fragmentation | Default in vLLM, SGLang, TensorRT-LLM, TGI |
| Scheduling | **Continuous (in-flight) batching** | GPU idle time | Iteration-level scheduling (Orca, vLLM) |
| Scheduling | **Chunked prefill** | Prefill stalls decode | Splits long prompts into chunks that share batches with decodes, which smooths ITL. |
| Reuse | **Prefix caching** (vLLM APC hashes blocks; SGLang **RadixAttention** uses a radix tree) | Repeated prefill | System prompts, RAG, agents, multi-turn chat. API "prompt caching" is the managed version. |
| Reuse | KV offload / tiering (CPU, SSD, remote), e.g. LMCache, Mooncake | HBM capacity | Keeps long or shared contexts warm across requests and nodes |
| Quantisation (weights) | FP8, INT8 (SmoothQuant), INT4 (AWQ, GPTQ), NVFP4/MXFP4 | Weight bytes, so decode speed | FP8 is near-lossless on Hopper. FP4 targets Blackwell. GGUF k-quants for llama.cpp. |
| Quantisation (KV) | **FP8 / INT8 / INT4 KV cache** (KIVI, KVQuant) | KV bytes | `--kv-cache-dtype fp8` in vLLM roughly halves KV, giving about 2× concurrency or context. |
| KV compression | Eviction / sparsity: StreamingLLM (sinks + window), H2O, SnapKV, PyramidKV, Quest | KV at very long T | Lossy. Validate on your own tasks. |
| Decoding | **Speculative decoding**: draft model, Medusa heads, **EAGLE-3**, **MTP** (DeepSeek-V3), n-gram / prompt lookup | Sequential decode | A draft proposes k tokens, the target verifies them in one pass, and rejection sampling keeps the output distribution *exact*. Typically 2–3× faster. |
| Parallelism | Tensor parallelism (within node), pipeline parallelism (across nodes), expert parallelism (MoE), data-parallel attention | Model > 1 GPU | TP splits matmuls, PP splits layers, EP splits experts. Context or sequence parallelism handles very long prompts. |
| System | **Disaggregated prefill/decode** (DistServe, Splitwise, Mooncake; NVIDIA Dynamo, llm-d) | Prefill/decode interference | Separate GPU pools, with the KV cache shipped over NVLink or RDMA. Tune TTFT and TPOT independently. |
| System | KV-aware routing / load balancing | Cache hit rate | Send requests to the replica that already holds their prefix. |
| Output | Structured / constrained decoding (XGrammar, Outlines, llguidance) | Invalid JSON / retries | Grammar masks applied to logits. Built into vLLM and SGLang. |

**Engines:**

| Engine | Strengths |
|---|---|
| **vLLM** | PagedAttention, continuous batching, APC, multi-LoRA, spec decode, FP8 KV, broad model support, V1 engine |
| **SGLang** | RadixAttention prefix tree, fast structured output, strong DeepSeek/MLA and large-scale EP serving |
| **TensorRT-LLM** | NVIDIA-optimised kernels, FP8/FP4, in-flight batching. Pairs with Triton / Dynamo. |
| **llama.cpp / Ollama** | CPU/Apple/edge, GGUF quantisation |
| HF TGI, LMDeploy, MLC-LLM | Production HF serving, TurboMind, cross-device compilation |

**Serving request lifecycle (vLLM):**

```
request → tokenizer → scheduler (waiting queue)
        → prefix-cache lookup (hash of block tokens + preceding prefix)
        → allocate free KV blocks → block table
        → [chunked] prefill: compute KV for uncached tokens, write to blocks
        → decode loop each iteration with other running seqs (continuous batching)
             FlashAttention/FlashInfer paged kernel reads KV via block table
             (+ speculative draft/verify)  → sampler → detokenise → stream
        → EOS / max_tokens → free blocks (or keep them as cached prefix)
```

### C3. Suggested learning path through this repo

1. Run S1 and S2. Recompute the table in A9 by hand for a model you actually use.
2. In S3, extend `mha.py` into a 2-layer model and write a cached `generate()`. Assert it matches `generate_naive`.
3. In S5 and S6, train tiny MHA/GQA/MLA models on the same text and compare loss at a **matched cache budget**
   (fix B5-1 first).
4. In S7, plug `apply_rotary(offset=past_len)` into MHA/GQA and test extrapolation with PI, NTK and YaRN.
5. On a GPU, run vLLM with `--enable-prefix-caching --kv-cache-dtype fp8` and a speculative config.
   Measure TTFT, TPOT and throughput against HF `generate()`. This is the missing `colab/` lab.
6. Fine-tune a QLoRA adapter with TRL or Unsloth, serve it through vLLM multi-LoRA, then try DPO or GRPO on top.

---

*Diagrams already in the repo: D01 curriculum · D02 naive vs cached · D03 MHA cache · D04 memory formula ·
D05 variant comparison · D06 GQA spectrum · D07 MLA compression · D08 RoPE rotation · D09 context
extension · D10 FlashAttention tiling · D11 SDPA dispatch · D12 PagedAttention blocks
(`docs/diagrams/`).*
