
Yes. These four concepts fit together through the **Roofline Model**. Let's use one concrete GPU and walk from **compute throughput → memory bandwidth → balanced arithmetic intensity → high/low AI → removing the bottleneck**.

---

# 1. Start with the two hardware limits

Assume a GPU has:

```text
Compute throughput = 100 TFLOPS
Memory bandwidth   = 500 GB/s
```

These are two different ceilings.

### Compute ceiling

The GPU can execute at most:

\[
\boxed{100\times10^{12}\ FLOPs/sec}
\]

### Memory ceiling

It can move at most:

\[
\boxed{500\times10^9\ Bytes/sec}
\]

from the relevant memory level.

Your application must operate underneath **both ceilings**.

---

# 2. Arithmetic intensity connects them

\[
\boxed{
AI=\frac{FLOPs}{Bytes}
}
\]

Suppose your workload has:

\[
AI=10\ FLOPs/Byte
\]

Then the memory system can feed:

\[
10\times500GB/s
\]

\[
=5000GFLOPS
\]

\[
=\boxed{5TFLOPS}
\]

So although the GPU is capable of:

```text
100 TFLOPS
```

your workload can only reach:

```text
5 TFLOPS
```

because **memory is the bottleneck**.

---

# 3. This gives the "balanced AI"

The **balanced arithmetic intensity**, often called the **ridge point**, is:

\[
\boxed
======

\frac{Compute\ Throughput}
{Memory\ Bandwidth}
}
\]

For our GPU:

\[
AI_
===

\frac{100TFLOPS}{500GB/s}
\]

\[
=\boxed{200\ FLOPs/Byte}
\]

This is the critical number.

---

# 4. Below 200 → memory bottleneck

Suppose:

\[
AI=50
\]

Memory can support:

\[
50\times500
===========

25,000GFLOPS
\]

\[
=\boxed{25TFLOPS}
\]

But GPU can compute:

\[
100TFLOPS
\]

Therefore:

```text
Memory ceiling = 25 TFLOPS
Compute ceiling = 100 TFLOPS
```

Actual theoretical ceiling:

\[
\boxed{25TFLOPS}
\]

You're **memory-bound**.

---

# 5. At 200 → balanced

Now:

\[
AI=200
\]

Memory can supply:

\[
200\times500
============

100,000GFLOPS
\]

\[
=\boxed{100TFLOPS}
\]

Exactly equal to the compute capability.

```text
Memory capability = 100 TFLOPS
Compute capability = 100 TFLOPS
```

This is the **balanced point**.

\[
\boxed{AI_{balanced}=200}
\]

---

# 6. Above 200 → compute bottleneck

Suppose:

\[
AI=500
\]

Memory could theoretically feed:

\[
500\times500
============

250TFLOPS
\]

But GPU only has:

\[
100TFLOPS
\]

Therefore:

\[
\boxed{100TFLOPS}
\]

is the ceiling.

Now the GPU's compute units are the bottleneck.

---

# 7. The entire thing in one table

|            AI | Memory-limited capability | Compute limit | Bottleneck         |
| ------------: | ------------------------: | ------------: | ------------------ |
|            10 |                  5 TFLOPS |           100 | **Memory**   |
|            50 |                 25 TFLOPS |           100 | **Memory**   |
|           100 |                 50 TFLOPS |           100 | **Memory**   |
| **200** |             **100** | **100** | **Balanced** |
|           500 |                       250 |           100 | **Compute**  |
|          1000 |                       500 |           100 | **Compute**  |

Therefore:

\[
\boxed{
Performance =
\min(
Compute\ Throughput,\;
AI\times MemoryBandwidth
)
}
\]

---

# 8. Roofline picture

Think of it like this:

```text
Performance
TFLOPS
  │
100│                         ─────────────────────
   │                       /
   │                     /
   │                   /
   │                 /
   │               /
   │             /
   │           /
   │         /
   │       /
   │     /
   │   /
   │ /
   └──────────────────────────────────────────→
                   AI
                 FLOPs/Byte

                  ↑
             Ridge point
               AI = 200
```

Left side:

```text
slope = memory bandwidth
```

Right side:

```text
flat = compute throughput
```

---

# 9. What does "high AI" actually mean?

High AI means:

> **You perform a lot of computation for every byte fetched from memory.**

Example:

```text
10,000 FLOPs
10 Bytes
```

\[
AI=1000
\]

That's high AI.

The GPU has plenty of work to keep its compute units busy.

Example workloads:

```text
Large GEMM
Matrix multiplication
Large Transformer MLP
Large-batch training
```

These can have high arithmetic intensity.

---

# 10. What does low AI mean?

Example:

```text
100 FLOPs
10 Bytes
```

\[
AI=10
\]

You're moving lots of data relative to the amount of computation.

Typical examples:

```text
Vector operations
Memory copies
Some reductions
LLM decode
KV-cache-heavy operations
```

These can become memory-bandwidth limited.

---

# 11. Now the important question: how do we remove a bottleneck?

There are **two completely different cases**.

---

# Case A — Memory bottleneck

Suppose:

```text
GPU:
100 TFLOPS
500 GB/s

Workload:
AI = 10
```

Maximum:

\[
10\times500=5TFLOPS
\]

You have:

```text
Compute capacity = 100
Actual ceiling   = 5
```

### Option 1 — Increase memory bandwidth

Suppose you upgrade:

```text
500 GB/s → 1000 GB/s
```

Then:

\[
10\times1000
============

10TFLOPS
\]

You doubled performance potential.

But you're still memory-bound.

---

# 12. Better option: reduce memory traffic

Suppose instead you change the algorithm:

```text
Before:

10 FLOPs / 1 Byte

After:

10 FLOPs / 0.25 Byte
```

Then:

\[
AI=40
\]

Memory ceiling:

\[
40\times500
===========

20TFLOPS
\]

You increased performance potential:

```text
5 TFLOPS → 20 TFLOPS
```

without changing the GPU.

This is why things like:

```text
FlashAttention
KV quantization
GQA
data reuse
kernel fusion
```

can be extremely valuable.

---

# 13. LLM example — remove memory bottleneck

Suppose decoding needs to read:

```text
1 GB KV data
```

for:

```text
10 GFLOPs
```

Then:

\[
AI=10
\]

Very low.

Now use FP8 KV instead of BF16.

Suppose KV traffic becomes:

```text
0.5 GB
```

while computation remains:

```text
10 GFLOPs
```

Then:

\[
AI=
\frac
=====

20
\]

You've doubled arithmetic intensity.

That means the memory-bound performance ceiling doubles.

---

# 14. Sliding window does something similar

Full context:

```text
128K KV
```

Sliding window:

```text
8K KV
```

Now you're moving dramatically less KV data per token.

So:

```text
Memory traffic ↓
AI ↑
TPOT ↓
```

assuming the workload remains otherwise comparable.

---

# 15. Case B — Compute bottleneck

Now suppose:

```text
AI = 1000
```

Memory bandwidth:

```text
500 GB/s
```

Memory could support:

\[
1000\times500
=============

500TFLOPS
\]

But GPU only has:

\[
100TFLOPS
\]

So:

\[
\boxed{100TFLOPS}
\]

is the bottleneck.

Now reducing memory traffic doesn't help much.

You need:

### Faster compute

```text
100 TFLOPS
      ↓
200 TFLOPS
```

or:

### Better algorithms

Reduce required FLOPs.

For example:

```text
1000 GFLOPs
      ↓
500 GFLOPs
```

through:

```text
lower precision
better algorithm
kernel optimization
sparsity
approximation
```

---

# 16. Tensor Cores are an example

Suppose a workload uses:

```text
FP32 CUDA cores
```

and gets:

```text
20 TFLOPS
```

Move to:

```text
BF16/FP16 Tensor Cores
```

and perhaps get:

```text
100 TFLOPS+
```

Now you've attacked the **compute ceiling**.

This is why Transformer training/inference uses:

```text
FP16
BF16
FP8
FP4
Tensor Cores
```

rather than relying purely on FP32.

---

# 17. But here's the subtle optimization strategy

You don't necessarily want:

> "Maximum AI."

You want:

> **Enough AI to reach the compute roofline, without unnecessary computation.**

Suppose:

```text
Ridge AI = 200
```

These two workloads:

### Workload A

\[
AI=200
\]

### Workload B

\[
AI=2000
\]

Both can potentially hit:

```text
100 TFLOPS
```

The second doesn't automatically mean it's 10× faster.

You've already reached the compute ceiling.

---

# 18. This is a very important optimization principle

```text
AI < Ridge point
       │
       ↓
Optimize memory movement
       │
       ↓
Increase AI
       │
       ↓
Reach ridge point
       │
       ↓
Now optimize computation
```

So:

### Memory-bound

Focus on:

```text
Memory bandwidth
↓
Data movement
↓
Cache reuse
↓
Fusion
↓
Compression
↓
Quantization
```

### Compute-bound

Focus on:

```text
Tensor Cores
↓
Precision
↓
FLOPs
↓
Kernel efficiency
↓
Parallelism
↓
Algorithmic complexity
```

---

# 19. Apply this to LLM inference

Now everything we've discussed comes together.

### Prefill

```text
Large prompt
     ↓
Large GEMMs
     ↓
High data reuse
     ↓
Higher AI
     ↓
Often compute-bound
```

Optimize with:

```text
Tensor Cores
FlashAttention
BF16/FP16/FP8
large efficient GEMMs
```

---

### Decode

```text
1 new token
     ↓
Q
     ↓
Read KV cache
     ↓
Attention
     ↓
next token
```

Often:

```text
Low AI
   ↓
Memory-bound
   ↓
TPOT bottleneck
```

Optimize with:

```text
GQA/MQA
KV quantization
Paged KV cache
sliding window
FlashAttention/optimized attention kernels
continuous batching
speculative decoding
```

---

# 20. And now your GPU comparison makes sense

Suppose:

```text
GPU A:
1000 TFLOPS
500 GB/s

GPU B:
500 TFLOPS
1000 GB/s
```

Which is faster?

**You cannot answer without knowing AI.**

### If AI = 10

GPU A:

\[
10\times500=5TFLOPS
\]

GPU B:

\[
10\times1000=10TFLOPS
\]

**GPU B has the higher memory-bound ceiling.**

### If AI = 500

GPU A:

\[
500\times500=250TFLOPS
\]

limited to:

\[
1000TFLOPS
\]

Actually GPU A's ceiling is 1000 TFLOPS, so 250 TFLOPS.

GPU B:

\[
500\times1000=500TFLOPS
\]

but its compute ceiling is 500 TFLOPS.

Both reach:

```text
GPU A → 250 TFLOPS
GPU B → 500 TFLOPS
```

Now compute throughput matters.

---

# 21. The architect-level mental model

When someone gives you:

```text
GPU:
X TFLOPS
Y TB/s
```

**don't immediately ask "how fast is it?"**

Ask:

### Step 1

\[
AI_{ridge}=\frac{X}{Y}
\]

### Step 2

Estimate workload AI:

\[
AI_{workload}=\frac{FLOPs}{Bytes}
\]

### Step 3

Compare:

```text
AIworkload < AIridge
       ↓
Memory-bound
```

or:

```text
AIworkload > AIridge
       ↓
Compute-bound
```

### Step 4

Optimize the appropriate bottleneck.

---

## The complete picture

```text
                         WORKLOAD
                            │
                            ↓
                   Arithmetic Intensity
                            │
                 ┌──────────┴──────────┐
                 │                     │
              LOW AI                HIGH AI
                 │                     │
                 ↓                     ↓
          MEMORY BOUND             COMPUTE BOUND
                 │                     │
                 ↓                     ↓
        Memory bandwidth          Compute throughput
        Data movement             Tensor Cores
        KV cache                  GEMM efficiency
        Cache reuse               Precision
        Quantization              FLOP reduction
                 │                     │
                 └──────────┬──────────┘
                            ↓
                     RIDGE POINT
                            │
                            ↓
                    MAX PERFORMANCE
```

### The one equation to remember

\[
\boxed}
=======

\min
\left(
P_{\text{compute}},
AI\times BW_{\text{memory}}
\right)
}
\]

And the optimization rule:

\[
\boxed{
AI<AI_{ridge}
\Rightarrow
\text{attack memory bottleneck}
}
\]

\[
\boxed{
AI>AI_{ridge}
\Rightarrow
\text{attack compute bottleneck}
}
\]

This is the fundamental reasoning behind **why FlashAttention, KV quantization, GQA, batching, Tensor Cores, FP8, and GPU selection** affect LLM performance differently.
Yes. This is the right next layer of understanding: **MHA → MQA → GQA → MLA → cross-layer sharing → hybrid attention** is essentially a progression of different ways to attack the **KV-cache + memory-bandwidth wall** while trying to preserve attention quality.

The key is to separate four objectives:

1. **Attention diversity / model quality**
2. **KV-cache memory**
3. **Arithmetic intensity / memory traffic**
4. **End-to-end inference efficiency**

---

# 1. Start from the real bottleneck

For autoregressive decoding, each new token needs:

\[
Q_t,\ K_t,\ V_t
\]

and then:

\[
O_t =
\operatorname{softmax}
\left(
\frac{Q_tK_{1:t}^{T}}{\sqrt{d_k}}
\right)V_{1:t}
\]

With KV caching, we don't recompute historical K/V.

But we still have to **read the cached K/V**.

So as context grows:

```text
Context
   │
   ▼
KV cache
   │
   │  read
   ▼
GPU attention kernel
   │
   ▼
next token
```

This creates the important inference problem:

\[
\boxed{
\text{Decode often becomes memory-bandwidth limited}
}
\]

The optimization question therefore becomes:

> **How can we store less information, move less information, or avoid storing it in every layer—without destroying the model's ability to retrieve information?**

That gives us the hierarchy below.

---

# 2. MHA — baseline

Multi-Head Attention:

\[
H_Q=H_K=H_V=H
\]

Suppose:

```text
Q = 32 heads
K = 32 heads
V = 32 heads
head_dim = 64
```

Each head has its own K/V:

```text
Q1 → K1,V1
Q2 → K2,V2
Q3 → K3,V3
...
Q32 → K32,V32
```

Therefore:

\[
M_
==

2LTHD\,S
\]

where:

- \(L\) = layers
- \(T\) = context
- \(H\) = heads
- \(D\) = head dimension
- \(S\) = bytes/element

### Advantage

Maximum K/V representation diversity.

Each head can learn:

```text
Head 1 → syntax
Head 2 → entity relation
Head 3 → long-range dependency
Head 4 → positional relation
...
```

### Disadvantage

Huge KV cache.

And during decode:

```text
GPU
 ↓
read K1,V1
read K2,V2
read K3,V3
...
read K32,V32
```

Large memory traffic.

---

# 3. MQA — attack H

Multi-Query Attention changes:

\[
H_{KV}:32\rightarrow1
\]

while keeping:

\[
H_Q=32
\]

So:

```text
             Q heads

Q1 ─┐
Q2 ─┤
Q3 ─┤
... ├────────→ Kshared
Q32─┘

             Vshared
```

The cache becomes:

\[
M_^
===

2L T(1)DS
\]

versus MHA:

\[
M_^
===

2LTHDS
\]

Therefore:

\[
\boxed}}
========

\frac{1}{H}
}
\]

For 32 heads:

\[
\boxed{32\times\ reduction}
\]

in the KV component.

---

# 4. But what happens to Attention Diversity?

This is the important part.

MQA does **not** reduce Q heads.

We still have:

\[
Q_1,Q_2,\ldots,Q_{32}
\]

Therefore:

\[
Q_1K^T
\neq
Q_2K^T
\]

generally.

So attention distributions can still be different:

```text
Q1 + Kshared → Attention pattern A

Q2 + Kshared → Attention pattern B

Q3 + Kshared → Attention pattern C
```

But all heads see the same K/V representation.

Therefore:

\[
\boxed{
\text{Q diversity remains}
}
\]

but:

\[
\boxed{
\text{K/V diversity collapses}
}
\]

This can cause quality degradation. The original GQA work explicitly found MQA can degrade quality and proposed GQA as an intermediate point. [arXiv](https://arxiv.org/abs/2305.13245?utm_source=chatgpt.com)

---

# 5. GQA — the sweet spot

GQA introduces:

\[
1 < H_{KV}<H_Q
\]

Example:

\[
H_Q=32,\qquad H_{KV}=8
\]

So:

```text
Q1 Q2 Q3 Q4 → KV1

Q5 Q6 Q7 Q8 → KV2

...

Q29 Q30 Q31 Q32 → KV8
```

Each group contains:

\[
\frac{32}{8}=4
\]

Q heads.

The important thing:

\[
Q_1\neq Q_2\neq Q_3\neq Q_4
\]

while:

\[
K_1=K_2=K_3=K_4
\]

within a group.

Therefore:

```text
             same K/V
                │
       ┌────────┼────────┐
       ↓        ↓        ↓
      Q1       Q2       Q3       Q4
       │        │        │        │
       ↓        ↓        ↓        ↓
    Attn A   Attn B   Attn C   Attn D
```

This is why GQA preserves much more attention diversity than simply collapsing everything to one KV head.

The original GQA paper showed that uptrained GQA can approach MHA quality while obtaining inference characteristics close to MQA. [arXiv](https://arxiv.org/abs/2305.13245?utm_source=chatgpt.com)

---

# 6. The memory equation exposes the entire game

The general equation:

\[
\boxed
======

2LBT H_{KV}D S
}
\]

Notice the variables.

```text
M_KV
 │
 ├── L  → number of layers
 ├── B  → batch
 ├── T  → sequence length
 ├── H  → KV heads
 ├── D  → head dimension
 └── S  → datatype
```

Different techniques attack different variables.

This is the most useful framework for understanding modern attention architectures.

---

# 7. GQA attacks \(H_\)

```text
MHA
H_KV = 32

        ↓

GQA
H_KV = 8

        ↓

MQA
H_KV = 1
```

So:

\[
\boxed{\text{GQA/MQA attack head dimension of the cache}}
\]

They don't change the fundamental \(L\times T\) scaling.

---

# 8. KV quantization attacks \(S\)

Suppose:

\[
S=2
\]

for FP16/BF16.

Use FP8:

\[
S=1
\]

Then:

\[
M_{KV}^{FP8}
\approx
\frac12M_{KV}^{FP16}
\]

So:

```text
FP16 KV
████████████

FP8 KV
██████
```

This is particularly attractive because the architecture remains essentially the same.

But lower precision introduces:

- quantization error
- scale management
- possible quality degradation
- kernel complexity

---

# 9. Sliding-window attention attacks \(T\)

Full attention:

\[
K_{1:T}
\]

Sliding window:

\[
K_{T-W+1:T}
\]

where:

\[
W\ll T
\]

So instead of:

```text
100K context
████████████████████████
```

the active attention context could be:

```text
8K window
        ████
```

Then local KV memory becomes approximately:

\[
M_{KV}\propto W
\]

rather than:

\[
M_{KV}\propto T
\]

But you sacrifice direct access to distant tokens.

---

# 10. MLA — a fundamentally different idea

Now we reach **DeepSeek's Multi-head Latent Attention**.

GQA says:

> "Let's store fewer K/V heads."

MLA says:

> **"Why store full K/V representations at all?"**

This is a deeper compression strategy.

DeepSeek-V2 introduced MLA specifically to compress KV cache into a latent representation; DeepSeek-V3 retained MLA as a central inference-efficiency mechanism. [arXiv](https://arxiv.org/abs/2405.04434?utm_source=chatgpt.com)

---

# 11. MHA/GQA cache

Normally we cache:

\[
K_t,V_t
\]

For every token.

Conceptually:

```text
Token
 │
 ├── K ───────────┐
 │                │
 └── V ───────────┤
                  ↓
              KV CACHE
```

MLA instead creates a compressed latent:

\[
c_t^{KV}=W^{DKV}h_t
\]

and caches:

\[
c_t^{KV}
\]

rather than explicitly storing the complete K/V tensors.

Conceptually:

```text
                 hidden state
                      │
                      ▼
              compression
                      │
                      ▼
                c_KV latent
                      │
                      ▼
                KV CACHE
```

The latent is much smaller.

---

# 12. Why "latent" works

Suppose full K/V contains:

```text
Large representation
████████████████████████████
```

MLA learns:

```text
Compressed representation
██████
```

Then attention projections can recover the information required for attention.

Mathematically, think:

\[
K \approx W^{UK}c^{KV}
\]

\[
V \approx W^{UV}c^{KV}
\]

where:

\[
c^{KV}=W^{DKV}h
\]

So instead of storing:

\[
K,V
\]

you store:

\[
\boxed{c^{KV}}
\]

plus a positional component.

---

# 13. Why the RoPE part matters

This is the subtle MLA architecture point.

You don't want to simply compress everything because positional information interacts with attention.

MLA therefore separates the content and positional components.

Conceptually:

```text
                 Hidden state
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
      Content path            Position path
          │                       │
       c_KV                    k_R
          │                       │
     compressed             RoPE applied
       latent                    │
          │                       │
          └───────────┬───────────┘
                      ▼
                   Attention
```

The shorthand you wrote:

\[
\boxed{d_c+d_R}
\]

is a useful way to remember the cache structure:

- \(d_c\): compressed latent/content dimension
- \(d_R\): decoupled RoPE positional dimension

So MLA is not simply "GQA with fewer heads."

It is:

> **A low-rank latent representation of the KV state + a separate positional pathway.**

---

# 14. MLA vs GQA

This is the important comparison.

### GQA

```text
32 Q heads
      ↓
8 KV heads
      ↓
store K/V
```

### MLA

```text
many Q heads
      ↓
compressed latent KV
      +
decoupled positional component
      ↓
store latent representation
```

So:

\[
\boxed{
GQA = head sharing
}
\]

while:

\[
\boxed{
MLA = representation compression
}
\]

This is a very important distinction.

---

# 15. MLA attacks \(H_\times D\)

GQA reduces:

\[
H_{KV}
\]

MLA effectively reduces the dimensionality of the information that must be stored:

\[
H_{KV}D
\rightarrow
d_c+d_R
\]

So the cache becomes approximately:

\[
M_{MLA}
\propto
LBT(d_c+d_R)S
\]

rather than:

\[
M_{GQA}
\propto
LBT H_{KV}D S
\]

That is why MLA can achieve extremely aggressive KV compression.

DeepSeek-V2 reported a 93.3% KV-cache reduction relative to the compared DeepSeek 67B baseline, along with higher generation throughput. [arXiv](https://arxiv.org/abs/2405.04434?utm_source=chatgpt.com)

---

# 16. Attention diversity: MHA → GQA → MLA

This is where you need to be precise.

### MHA

Maximum explicit K/V diversity:

\[
K_1,K_2,\ldots,K_H
\]

### GQA

Some K/V sharing:

\[
K_1=K_2=K_3=K_4
\]

within a group.

But Q remains independent.

### MLA

It doesn't primarily say:

> "Let's use fewer KV heads."

Instead:

> **"Let's learn a compact latent representation from which the attention projections can be generated."**

Therefore, MLA is a **learned low-rank compression problem**, rather than simple head sharing.

This can preserve more useful attention structure for a given cache budget than aggressively reducing KV heads.

---

# 17. Now attack \(L\): Cross-Layer Attention

This is another beautiful idea.

Look at your KV equation:

\[
M_{KV}\propto L
\]

Suppose:

\[
L=64
\]

Even if GQA reduces KV heads, you still have:

```text
Layer 1  → KV
Layer 2  → KV
Layer 3  → KV
...
Layer 64 → KV
```

So:

\[
64\times KV
\]

What if adjacent layers share KV?

That's **Cross-Layer Attention (CLA)**.

Instead of:

```text
Layer 1 → KV1
Layer 2 → KV2
Layer 3 → KV3
Layer 4 → KV4
```

you can have:

```text
Layer 1 ─┐
         ├── KV1
Layer 2 ─┘

Layer 3 ─┐
         ├── KV2
Layer 4 ─┘
```

Now the cache is approximately halved.

The CLA paper reports another ~2× KV-cache reduction relative to MQA in its experiments while maintaining nearly the same accuracy as its MQA baseline. [arXiv](https://arxiv.org/abs/2405.12981?utm_source=chatgpt.com)

---

# 18. The attack dimensions are now obvious

This is an excellent architecture mental model:

\[
\boxed
======

2LBT H_{KV}DS
}
\]

Different methods attack different terms:

| Method               | Attacks                     | Main idea                                |
| -------------------- | --------------------------- | ---------------------------------------- |
| MHA                  | —                          | Full diversity                           |
| GQA                  | \(H_{KV}\)                  | Group KV heads                           |
| MQA                  | \(H_{KV}\)                  | One KV head                              |
| KV quantization      | \(S\)                       | Fewer bytes                              |
| Sliding window       | \(T\)                       | Keep local context                       |
| MLA                  | \(H_{KV}D\) effectively     | Compress KV representation               |
| CLA                  | \(L\)                       | Share KV across layers                   |
| YOCO                 | \(L\) / cache structure     | Cache global KV once                     |
| SSM/linear attention | \(T\) scaling/state         | Replace full KV retrieval in many layers |
| Hybrid attention     | fraction of layers using KV | Only some layers use full attention      |

This is the framework I'd remember.

---

# 19. YOCO — "cache once"

YOCO goes even further.

Instead of every decoder layer independently maintaining its own KV cache:

```text
Layer 1 → KV
Layer 2 → KV
Layer 3 → KV
...
Layer 32 → KV
```

YOCO uses a self-decoder to build global KV representations that can be reused by another decoder component through cross-attention.

Conceptually:

```text
              Input
                │
                ▼
         Self-decoder
                │
                ▼
        Global KV cache
                │
       ┌────────┼────────┐
       ↓        ↓        ↓
    Cross     Cross    Cross
    decoder   decoder  decoder
       │        │        │
       └────────┼────────┘
                ↓
             Output
```

Hence:

> **You Only Cache Once.**

The YOCO paper explicitly targets reducing GPU memory by caching KV once while retaining global attention capability. [arXiv](https://arxiv.org/abs/2405.05254?utm_source=chatgpt.com)

---

# 20. Hybrid SSM / linear attention

Now we attack the problem differently.

Instead of asking:

> "How can I compress the KV cache?"

we ask:

> **"Can I avoid a conventional KV cache in most layers?"**

This is the idea behind hybrid architectures such as Jamba and newer architectures such as Qwen3-Next.

---

# 21. Transformer attention vs SSM

Standard attention:

\[
O_t=
softmax(Q_tK_{1:t}^{T})V_{1:t}
\]

requires access to historical tokens.

SSM/linear attention instead maintains a recurrent state:

\[
s_t=f(s_{t-1},x_t)
\]

Conceptually:

```text
Token 1 → State 1
             ↓
Token 2 → State 2
             ↓
Token 3 → State 3
             ↓
...
```

Instead of:

```text
Token 1 ─┐
Token 2 ─┤
Token 3 ─┤
...      ├──→ huge KV cache
Token T ─┘
```

you maintain:

```text
Token stream
     ↓
small recurrent state
```

This can dramatically reduce long-context memory pressure.

---

# 22. But pure SSM has a problem

A recurrent state is a **compressed summary**.

Suppose the model needs to retrieve exactly:

> "What was the fifth number in paragraph 72?"

Full attention can directly retrieve:

\[
K_{72,5}
\]

A compressed recurrent state has to retain that information in its state.

So there is a fundamental trade-off:

```text
Full attention
   ↓
Exact token-level retrieval
   ↓
Expensive KV memory

SSM / linear attention
   ↓
Compressed recurrent memory
   ↓
Efficient
   ↓
Potentially weaker arbitrary retrieval
```

That's why hybrids are attractive.

---

# 23. Jamba

Jamba combines Transformer attention and Mamba layers rather than using only one architecture. The original design used a relatively small fraction of Transformer attention layers, with Mamba handling most of the sequence processing. [arXiv](https://arxiv.org/abs/2403.19887?utm_source=chatgpt.com)

Conceptually:

```text
Layer 1   Mamba
Layer 2   Mamba
Layer 3   Mamba
Layer 4   Mamba
Layer 5   Mamba
Layer 6   Mamba
Layer 7   Mamba
Layer 8   Attention
Layer 9   Mamba
...
```

So most layers don't maintain the same kind of full KV cache.

---

# 24. Qwen3-Next

Qwen3-Next pushes this idea further with a hybrid of **Gated DeltaNet + Gated Attention**.

Its published architecture uses a roughly:

\[
3:1
\]

ratio:

```text
3 × Gated DeltaNet
        +
1 × Gated Attention
```

So approximately 75% of layers use the recurrent/linear-style mechanism and 25% retain full attention. [Qwen](https://qwen.ai/blog?id=qwen3-next\&utm_source=chatgpt.com)

The idea is beautiful:

```text
                 48 layers

DeltaNet  DeltaNet  DeltaNet  Attention
    ↓         ↓         ↓          ↓
   cheap     cheap     cheap     retrieval

DeltaNet  DeltaNet  DeltaNet  Attention
    ↓         ↓         ↓          ↓
```

The recurrent layers provide efficient long-context processing.

The full-attention layers preserve strong direct retrieval.

---

# 25. This creates an "attention budget"

This is a useful way to think about modern architectures.

You don't necessarily need:

\[
100\%
\]

full attention.

Maybe:

\[
25\%
\]

of layers can provide the expensive high-fidelity retrieval mechanism.

Then:

\[
75\%
\]

can use cheaper state-based processing.

So architecture becomes:

\[
\boxed{
\text{Capacity} =
\text{cheap memory}
+
\text{selective exact retrieval}
}
\]

Qwen's own explanation describes this motivation: linear attention is efficient but has recall limitations, while standard attention has stronger direct retrieval but higher inference cost; Qwen3-Next mixes them to balance the two. [Qwen](https://qwen.ai/blog?id=qwen3-next\&utm_source=chatgpt.com)

---

# 26. Now bring in Arithmetic Intensity

This is where your previous **memory wall** discussion connects.

Recall:

\[
AI=\frac{FLOPs}{Bytes}
\]

and:

\[
P_
==

\min(P_{compute},AI\times BW)
\]

For decode, a major cost is:

```text
Read KV
   ↓
Attention
```

If:

\[
Bytes\gg FLOPs
\]

then:

\[
AI\downarrow
\]

and you become memory-bound.

---

# 27. What GQA does to Arithmetic Intensity

Suppose:

```text
MHA:
32 KV heads
```

versus:

```text
GQA:
8 KV heads
```

GQA reduces KV bytes transferred.

So:

\[
Bytes_{GQA}<Bytes_{MHA}
\]

For approximately similar attention computation:

\[
AI_
===

\frac{FLOPs}{Bytes_{GQA}}
\]

Therefore:

\[
\boxed{
AI_{GQA}>AI_{MHA}
}
\]

in the KV-read component.

This is an important insight:

> **GQA isn't only a VRAM optimization. It can also improve arithmetic intensity by reducing memory traffic.**

---

# 28. MLA pushes this further

MLA reduces the amount of cached representation that needs to be stored/read.

So:

\[
Bytes_{MLA}\ll Bytes_{MHA}
\]

which can raise effective arithmetic intensity:

\[
AI_
===

\frac{FLOPs}{Bytes_{MLA}}
\]

Potentially moving decode closer to the compute roof.

That's why these architectures matter to **GPU utilization**, not just memory capacity.

---

# 29. But don't confuse "higher AI" with "faster automatically"

This is an important engineering point.

Suppose:

\[
P_{compute}=1000\,TFLOPS
\]

and:

\[
BW=1.8\,TB/s
\]

Then:

\[
AI_
===

\frac{1000\times10^{12}}
{1.8\times10^{12}}
\approx556
\]

So if your workload has:

\[
AI=50
\]

then:

\[
P_
==

50\times1.8
===========

90TFLOPS
\]

Even though the GPU has:

\[
1000TFLOPS
\]

of compute capability.

You're still memory-bound.

---

# 30. Therefore modern attention optimization has two objectives

### Objective 1 — Fit

\[
M_{KV}\le M_{VRAM}
\]

If it doesn't fit:

```text
OOM
```

### Objective 2 — Move less

Even if it fits:

```text
VRAM
 ↓
memory bandwidth
 ↓
GPU cores
```

can still be the latency bottleneck.

So:

\[
\boxed{
\text{Memory capacity} \neq
\text{Memory bandwidth}
}
\]

GQA/MLA help both.

---

# 31. Full architecture evolution

This is the mental map I recommend memorizing:

```text
                         KV CACHE PROBLEM
                               │
                               ▼
                    M_KV ∝ L × T × H_KV × D
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
          Reduce H          Reduce L          Reduce T
             │                 │                 │
        ┌────┴────┐       ┌────┴────┐      ┌────┴────┐
        │         │       │         │      │         │
       GQA       MQA     CLA       YOCO   Window   SSM
        │         │       │         │      │         │
       8 KV      1 KV    share     cache   local    state
       heads     head    layers    once    context
             │
             ▼
            MLA
       compress KV state
             │
             ▼
        latent cache
```

And then:

```text
Hybrid architectures
       │
       ├── Attention layers
       │      ↓
       │   exact retrieval
       │
       └── SSM / linear layers
              ↓
           cheap state
```

---

# 32. Attention diversity vs efficiency

You can visualize the engineering trade-off as:

```text
                         QUALITY /
                  ATTENTION DIVERSITY
                          ↑
                          │
              MHA ●       │
                          │
                   GQA ●  │
                          │
                  MLA ●   │
                          │
              MQA ●       │
                          │
             Hybrid ●     │
                          │
                          └──────────────────→
                              efficiency
```

But **MLA doesn't simply sit between MHA and MQA on a one-dimensional "diversity" axis**. It changes the representation being cached rather than merely reducing head count.

That's an important conceptual distinction.

---

# 33. Retrofit / conversion

Now suppose you already have:

```text
Llama-style MHA
```

and you don't want to train a new model from scratch.

There are two important approaches.

### MHA → GQA

Group the original heads:

\[
K_g \approx \frac{1}{n}\sum_{i\in g}K_i
\]

and similarly:

\[
V_g \approx \frac{1}{n}\sum_{i\in g}V_i
\]

Then uptrain.

The GQA paper demonstrated an uptraining recipe from existing MHA checkpoints using only about 5% of the original pretraining compute. [arXiv](https://arxiv.org/abs/2305.13245?utm_source=chatgpt.com)

This is why GQA is practical for retrofitting.

---

# 34. MHA → MLA

Harder.

You need to approximate:

\[
K \approx W_{UK}c
\]

\[
V \approx W_{UV}c
\]

with:

\[
c=W_Dh
\]

This is effectively a low-rank factorization/compression problem.

Recent MHA→MLA conversion work uses techniques such as joint SVD and partial-RoPE handling; one reported Llama-2-7B conversion reduced KV-cache size by 92.19% with a reported ~0.5-point LongBench drop after adaptation. [arXiv](https://arxiv.org/abs/2502.14837?utm_source=chatgpt.com)

So:

```text
MHA
 ↓
low-rank approximation
 ↓
RoPE restructuring
 ↓
fine-tuning / adaptation
 ↓
MLA
```

This is substantially more invasive than GQA conversion.

---

# 35. Why retraining cost matters

This is the part often missed in "SOTA architecture" discussions.

You don't get efficiency for free.

There are three levels:

### Level 1 — Serving optimization

No model retraining:

```text
PagedAttention
FlashAttention
KV quantization
continuous batching
prefix caching
```

Cheap to adopt.

---

### Level 2 — Architecture compression

Requires model adaptation:

```text
MHA → GQA
MHA → MQA
MHA → MLA
```

Requires some retraining/uptraining.

---

### Level 3 — Fundamental architecture change

Potentially substantial training:

```text
Transformer
   ↓
Hybrid attention/SSM
```

or:

```text
Full KV layers
   ↓
Cross-layer sharing
```

This changes the learned computation graph.

---

# 36. Your "Inference Efficiency Engineering" stack

For an AI Architect, I would organize all this into **five layers**:

```text
                 INFERENCE EFFICIENCY
                         │
 ┌───────────────────────┼────────────────────────┐
 │                       │                        │
 ▼                       ▼                        ▼
MODEL ARCHITECTURE    CACHE DESIGN           KERNEL
 │                       │                        │
 ├── GQA                 ├── Paged KV            ├── FlashAttention
 ├── MQA                 ├── Prefix cache         ├── FlashMLA
 ├── MLA                 ├── KV quantization     ├── fused kernels
 ├── CLA                 └── offloading          └── Tensor Cores
 └── Hybrid SSM
 │
 ▼
RUNTIME / SCHEDULING
 │
 ├── continuous batching
 ├── speculative decoding
 ├── prefill/decode disaggregation
 ├── tensor parallelism
 └── expert parallelism
 │
 ▼
HARDWARE
 │
 ├── VRAM capacity
 ├── HBM/GDDR bandwidth
 ├── Tensor Core throughput
 ├── NVLink
 └── PCIe
```

---

# 37. The most important equation for your notes

Write this next to your KV-cache diagram:

\[
\boxed
======

2LBT H_{KV}D S
}
\]

Then underneath:

```text
GQA / MQA
      ↓
reduce H_KV

KV Quantization
      ↓
reduce S

Sliding Window
      ↓
reduce T

CLA / YOCO
      ↓
reduce effective L

MLA
      ↓
compress H_KV × D
into latent representation

SSM / Linear Attention
      ↓
replace token-level KV retrieval
with recurrent state

Hybrid Attention
      ↓
use full KV only where exact retrieval matters
```

And finally connect it to the roofline:

\[
\boxed{
AI=\frac{FLOPs}{Bytes}
}
\]

\[
\boxed{
P=\min(P_{compute},AI\times BW)
}
\]

Therefore:

> **Modern LLM inference engineering is largely the art of reducing the amount of information that must be moved through the memory hierarchy without removing the information the model needs for high-quality retrieval.**

That's the common principle behind **GQA → MLA → CLA/YOCO → hybrid attention/SSM**.

For your AI-Architect preparation, this is the strongest mental model: **GQA attacks head redundancy, MLA attacks representation redundancy, CLA/YOCO attack layer redundancy, and SSM/hybrid architectures attack the assumption that every layer needs an explicit token-level KV cache at all.**

Exactly. Let's take **A100 40 GB** and build the complete VRAM budget around an LLM.

One important distinction first:

> **40 GB VRAM is the physical capacity. You cannot allocate all 40 GB to model weights + KV cache.** You need runtime/workspace/headroom as well.

A useful engineering approximation is:

\[
\boxed{
VRAM =
W_{model}
+
KV_{cache}
+
Activations
+
CUDA/workspace
+
Runtime/allocator
}
\]

---

# A100 40 GB — LLM inference memory

Conceptually:

```text
A100 40 GB
┌──────────────────────────────────────────────┐
│                                              │
│  Model Weights                               │
│  ██████████████████████████                  │
│                                              │
│  KV Cache                                    │
│  ████████████                                │
│                                              │
│  Activations / temporary tensors             │
│  ███                                         │
│                                              │
│  CUDA / attention workspace                  │
│  ██                                          │
│                                              │
│  Runtime / allocator / headroom              │
│  ██                                          │
│                                              │
└──────────────────────────────────────────────┘
                    40 GB
```

The exact proportions are workload-dependent.

---

# 1. Model weights

Suppose the model has:

\[
N=7B
\]

parameters.

For FP16/BF16:

\[
2\ bytes/parameter
\]

Therefore:

\[
7B\times2
=========

14GB
\]

So roughly:

\[
\boxed{W_{model}\approx14GB}
\]

for a 7B model.

For comparison:

| Model | FP16/BF16 weights |
| ----: | ----------------: |
|    7B |            ~14 GB |
|   13B |            ~26 GB |
|   20B |            ~40 GB |
|   30B |            ~60 GB |
|   70B |           ~140 GB |

Therefore a **70B FP16 model cannot fit on one A100 40GB**.

You need tensor parallelism, quantization, or multiple GPUs.

---

# 2. KV cache

Now use your equation:

\[
\boxed
======

2LBT H_{KV}DS
}
\]

Suppose:

```text
Layers       = 32
Q heads      = 32
KV heads     = 8
Head dim     = 64
Batch        = 1
Context      = 100K
FP16         = 2 bytes
```

Then:

\[
M_
==

2(32)(1)(100000)(8)(64)(2)
\]

≈

\[
\boxed{6.55GB}
\]

So:

```text
7B FP16 model
       ↓
~14 GB weights

100K context
       ↓
~6.55 GB KV
```

Already:

\[
14+6.55
=======

20.55GB
\]

used.

---

# 3. What about the remaining ~19.5 GB?

It isn't simply "free".

You need:

### Activations

During inference, intermediate tensors exist.

During decode they are relatively small compared with the weights/KV cache, especially for batch 1, but with large batches and long prompts they matter.

---

### CUDA workspace

Kernels can require temporary buffers.

For example:

```text
FlashAttention
GEMM
cuBLAS
quantization
communication
```

may require workspace.

---

### Attention workspace

Depending on the implementation:

```text
Q
K
V
temporary buffers
softmax-related buffers
```

are needed.

FlashAttention reduces the need to materialize a huge \(T\times T\) attention matrix, which is important for long-context prefill.

---

### Runtime / allocator

Frameworks such as PyTorch and inference engines need memory for:

```text
CUDA context
memory allocator
kernel metadata
buffers
graphs
scheduler
```

---

# 4. Therefore don't think:

```text
40 GB
-
14 GB weights
=
26 GB KV
```

Instead think:

\[
\boxed
======

VRAM
----

Weights
-------

Workspace
---------

Runtime
-------

Safety\ margin
}
\]

For example, if you reserve approximately 4–6 GB for everything else:

```text
A100 40 GB

40 GB
│
├── 14 GB   Model
│
├── ~6 GB   KV
│
├── ~4 GB   Runtime/workspace
│
└── ~16 GB  potentially available for more KV/batching
```

The actual allocation strategy depends on the serving engine.

---

# 5. Now see why GQA matters

Suppose the same model used **MHA**:

\[
H_{KV}=32
\]

instead of:

\[
H_{KV}=8
\]

KV becomes:

\[
6.55\times\frac{32}{8}
\]

\[
=\boxed{26.2GB}
\]

Now:

```text
A100 40GB

Model weights     14 GB
KV cache          26.2 GB
────────────────────────
                   40.2 GB
```

You've already exceeded the GPU before accounting for runtime/workspace.

So **100K context is impossible in this configuration**.

But with GQA:

```text
Model = 14 GB
KV    = 6.55 GB
```

you have much more room.

That's the real-world value of GQA.

---

# 6. Now compare MHA / GQA / MQA on A100

For your hypothetical 32-layer architecture:

```text
Context = 100K
FP16
D = 64
```

| Architecture  |    KV heads |           KV cache |         Weight + KV |
| ------------- | ----------: | -----------------: | ------------------: |
| MHA           |          32 |           ~26.2 GB |            ~40.2 GB |
| GQA           |          16 |           ~13.1 GB |            ~27.1 GB |
| **GQA** | **8** | **~6.55 GB** | **~20.55 GB** |
| GQA           |           4 |           ~3.28 GB |           ~17.28 GB |
| MQA           |           1 |           ~0.82 GB |           ~14.82 GB |

Ignoring workspace/runtime for simplicity.

This shows exactly why:

\[
\boxed{
H_{KV}
}
\]

is such an important architectural choice.

---

# 7. Batch makes A100 memory disappear very quickly

Suppose your GQA model has:

\[
KV_{1seq}=6.55GB
\]

At batch 1:

\[
6.55GB
\]

Batch 2:

\[
13.1GB
\]

Batch 4:

\[
26.2GB
\]

Batch 8:

\[
52.4GB
\]

So:

```text
100K context + GQA

B=1   ██████                 6.55 GB
B=2   ████████████          13.1 GB
B=4   ████████████████████  26.2 GB
B=8   ██████████████████████████████ 52.4 GB
```

That's why production serving doesn't simply use:

> "Batch = 32"

and hope for the best.

The scheduler has to account for **actual sequence lengths and KV-cache allocation**.

---

# 8. A100 architecture connection

The A100 has:

- **40 GB HBM2** on the 40GB version
- very high HBM bandwidth
- Tensor Cores
- large compute throughput

But don't confuse:

```text
40 GB
```

with:

```text
40 GB/s
```

or:

```text
TFLOPS
```

They are different.

### VRAM capacity

\[
\boxed{40GB}
\]

Answers:

> How much data can I keep resident?

### HBM bandwidth

Answers:

> How quickly can I move that data?

### Tensor Core throughput

Answers:

> How quickly can I perform matrix operations once data is available?

---

# 9. This gives you the complete LLM bottleneck picture

```text
                  A100
                   │
       ┌───────────┼────────────┐
       │           │            │
       ▼           ▼            ▼
   VRAM capacity  HBM BW    Tensor Cores
       │           │            │
       │           │            │
    "Can it fit?"  │       "Can I compute?"
                   │
             "Can I feed
              the cores?"
```

During **prefill**:

```text
Large matrix operations
        ↓
High arithmetic intensity
        ↓
Tensor Cores heavily used
        ↓
Often compute-bound
```

During **decode**:

```text
One new token
     ↓
Read KV cache
     ↓
Attention
     ↓
Next token
```

Often:

```text
Large memory movement
        ↓
Lower arithmetic intensity
        ↓
Memory-bandwidth pressure
        ↓
GPU cores may WAIT
```

---

# 10. This is why you optimize in this order

For a real A100 inference deployment:

### Step 1 — Can the weights fit?

\[
W_{model}\le VRAM
\]

If not:

```text
quantization
tensor parallelism
multi-GPU
```

---

### Step 2 — Can the KV cache fit?

\[
W_{model}+KV\le VRAM
\]

If not:

```text
GQA/MQA
MLA
KV quantization
sliding window
cross-layer sharing
```

---

### Step 3 — Can you achieve required TPOT?

If KV fits but decode is slow:

```text
KV bandwidth
 ↓
GQA
KV quantization
FlashAttention/optimized kernels
PagedAttention
continuous batching
```

---

### Step 4 — Can you increase throughput?

Then:

```text
continuous batching
prefix caching
speculative decoding
better scheduling
tensor parallelism
```

---

# 11. The most useful A100 mental model

Write this:

```text
                 A100 40 GB
                     │
        ┌────────────┴────────────┐
        │                         │
    VRAM capacity             Compute
        │                         │
        ▼                         ▼
  ┌──────────────┐         ┌─────────────┐
  │ Model weights│         │ Tensor Cores│
  │ KV cache     │         │ CUDA cores  │
  │ Activations  │         │             │
  │ Workspace    │         └─────────────┘
  └──────────────┘
        │
        ▼
   HBM bandwidth
        │
        ▼
    Feed compute
```

And the architectural equation:

\[
\boxed
======

M_{weights}
+
2LBT H_{KV}DS
+
M_{activations}
+
M_{workspace}
+
M_{runtime}
}
\]

Then the performance equation:

\[
\boxed
======

\min(P_{compute},AI\times BW_{HBM})
}
\]

So for your **AI Architect** mental model:

> **Model weights determine the baseline VRAM footprint. KV cache determines how that footprint grows with context and concurrency. GQA/MLA reduce the KV component. HBM bandwidth determines how quickly the cached state can be fed to the compute units. Tensor Cores determine how quickly the arithmetic can be performed. The latency bottleneck is whichever side—memory movement or computation—reaches its roof first.**

Yes. If we frame the problem from an **AI Architect / LLM inference architecture** perspective, the problems around **model architecture + GPU + KV-cache management** can be organized into a few fundamental bottlenecks.

The central equation is:

\[
\boxed{
M_{KV}=2\times L\times B\times T\times H_{KV}\times D\times S
}
\]

where:

- \(L\) = number of layers
- \(B\) = concurrent sequences
- \(T\) = cached context tokens
- \(H_{KV}\) = number of KV heads
- \(D\) = head dimension
- \(S\) = bytes per KV element
- 2 = K + V

---

# 1. The fundamental problems

Think of the entire inference system as:

```text
                LLM Inference
                     │
        ┌────────────┼────────────┐
        │            │            │
     Model         GPU          Runtime
   Architecture   Hardware      / Serving
        │            │            │
        ▼            ▼            ▼
   KV size       VRAM limit    KV allocation
   KV traffic    Bandwidth     batching
   FLOPs         Compute       scheduling
   quality       utilization   cache reuse
```

There are **three major constraints**:

### A. Can the model fit?

\[
Weights + KV + Activations + Runtime < VRAM
\]

### B. Can the GPU feed the compute units?

\[
AI = \frac{FLOPs}{Bytes}
\]

If AI is low, you hit the **memory wall**.

### C. Can we maintain quality while reducing memory?

This is where:

- MQA
- GQA
- MLA
- CLA
- YOCO
- SSM/linear attention
- KV quantization

come in.

---

# 2. Architecture problems

## Problem 1 — MHA creates huge KV cache

Traditional MHA:

\[
H_Q=H_K=H_V
\]

For example:

```text
32 Query heads
32 Key heads
32 Value heads
```

KV cache:

\[
M_{KV}\propto H_{KV}
\]

So with 32 KV heads:

```text
Q1 ─ K1 V1
Q2 ─ K2 V2
...
Q32 ─ K32 V32
```

Every head has its own K/V history.

### Problem

At long context:

```text
4K       → manageable
32K      → large
100K     → very large
1M       → enormous
```

And with batching:

```text
1 request
     ↓
8 requests
     ↓
32 requests
     ↓
128 requests
```

KV grows **linearly with batch**.

---

# 3. GQA/MQA solve only one dimension

GQA:

```text
32 Q heads

Q1 Q2 Q3 Q4 → KV1
Q5 Q6 Q7 Q8 → KV2
...
```

For 32 Q heads and 8 KV heads:

\[
r=\frac{32}{8}=4
\]

This reduces:

\[
H_{KV}:32\rightarrow8
\]

Therefore:

\[
M_{KV}\downarrow4\times
\]

But the other terms remain:

\[
L,\ B,\ T,\ D,\ S
\]

So GQA **doesn't solve everything**.

---

# 4. Attention diversity problem

There is a quality trade-off.

MHA:

```text
Q1 → K1,V1
Q2 → K2,V2
Q3 → K3,V3
...
Q32 → K32,V32
```

Maximum K/V representation diversity.

GQA:

```text
Q1 ─┐
Q2 ─┤
Q3 ─┤→ KV1
Q4 ─┘

Q5 ─┐
Q6 ─┤
Q7 ─┤→ KV2
Q8 ─┘
```

The Q heads remain independent:

\[
Q_i \neq Q_j
\]

but K/V representations are shared.

So:

> **GQA preserves Query-side attention diversity but reduces Key/Value representation diversity.**

This creates the architecture problem:

\[
\boxed{
Memory\ efficiency
\leftrightarrow
Attention\ representation\ diversity
}
\]

---

# 5. MQA pushes sharing too far

MQA:

\[
H_{KV}=1
\]

```text
Q1 ─┐
Q2 ─┤
Q3 ─┤
...
Q32─┘
     ↓
   KV shared
```

Huge memory savings.

But now all Q heads see the same K/V representation.

The Q heads can still produce different attention distributions:

\[
A_i=softmax(Q_iK^T)
\]

but the underlying K/V representation is shared.

So:

```text
MHA
 ↓
maximum KV diversity
high memory

GQA
 ↓
moderate sharing
moderate memory

MQA
 ↓
maximum sharing
minimum memory
```

---

# 6. GQA ratio selection itself is a problem

Suppose:

\[
H_Q=32
\]

Possible choices:

| KV heads | Sharing ratio |
| -------: | ------------: |
|       32 |       1:1 MHA |
|       16 |           2:1 |
|        8 |           4:1 |
|        4 |           8:1 |
|        1 |      32:1 MQA |

There isn't a universal optimal ratio.

You have to balance:

\[
Quality
\leftrightarrow
KV\ memory
\leftrightarrow
Bandwidth
\leftrightarrow
Latency
\]

That's an architecture search problem.

---

# 7. KV cache grows with context

Even with GQA:

\[
M_{KV}\propto T
\]

Suppose:

```text
32 layers
8 KV heads
64 head dimension
BF16
```

KV cost is approximately:

```text
64 KiB/token
```

Therefore:

| Context |       KV |
| ------: | -------: |
|      4K | ~256 MiB |
|      8K | ~512 MiB |
|     16K |   ~1 GiB |
|     32K |   ~2 GiB |
|    100K | ~6.1 GiB |
|      1M |  ~61 GiB |

So even after GQA:

> **Long context becomes a memory-capacity problem.**

---

# 8. Batch makes KV worse

This is particularly important for production serving.

If:

\[
KV_{1request}=6.1GB
\]

then:

```text
1 request  → 6.1 GB
4 requests → 24.4 GB
8 requests → 48.8 GB
```

Suddenly a GPU that could handle one long-context request cannot handle many concurrent requests.

So:

\[
\boxed{
KV\ cache\ is\ a\ concurrency\ problem
}
\]

not merely a context-length problem.

---

# 9. GPU VRAM capacity problem

The GPU has finite VRAM.

For example:

```text
A100 40GB
```

Your actual budget is:

\[
40GB =
Weights+
KV+
Activations+
Workspace+
Runtime+
Fragmentation
\]

Not:

\[
40GB=Weights+KV
\]

So if:

```text
Weights     = 14 GB
KV          = 10 GB
Workspace   = 3 GB
Runtime     = 2 GB
Activation  = 2 GB
------------------
Total       = 31 GB
```

you don't really have 9 GB freely available in practice.

---

# 10. Model weights compete with KV

This is an important architectural conflict.

Suppose:

```text
7B FP16
≈14 GB weights
```

Then:

```text
A100 40GB

14 GB → weights
 6 GB → KV
 ? GB → activations
 ? GB → CUDA
 ? GB → workspace
 ? GB → batching
```

Long context or higher concurrency can consume the remaining VRAM.

Therefore:

\[
\boxed{
Model\ size
\leftrightarrow
Context
\leftrightarrow
Concurrency
}
\]

are competing for the same physical resource.

---

# 11. GPU memory bandwidth problem

Even if KV fits in VRAM, you still have another problem.

During decoding:

```text
GPU
 │
 ├── Q
 │
 ├── Read KV cache
 │      ↓
 │   HBM/VRAM
 │
 ├── Attention
 │
 └── next token
```

Every generated token needs to read historical K/V.

So:

\[
Bytes_{KV}\uparrow
\]

and memory traffic increases.

This can make decode:

\[
\boxed{\text{memory-bandwidth bound}}
\]

rather than compute-bound.

---

# 12. Tensor Core utilization can be misleading

You might see:

```text
GPU utilization = 35%
```

and think:

> GPU is poorly utilized.

Not necessarily.

If the workload is memory-bound:

```text
Memory
  ↓
KV
  ↓
Tensor Cores waiting
```

The Tensor Cores cannot work faster because data isn't arriving fast enough.

Roofline:

\[
P=\min(P_{compute}, AI\times BW)
\]

If:

\[
AI\times BW \ll P_{compute}
\]

then memory is the bottleneck.

---

# 13. Increasing GPU compute doesn't necessarily solve decode

Suppose you upgrade:

```text
GPU A → GPU B
```

and get:

\[
2\times Tensor\ FLOPS
\]

but only:

\[
1.2\times memory\ bandwidth
\]

For a memory-bound decode workload:

```text
2× compute
       ↓
almost no proportional
decode improvement
```

because:

\[
Performance\approx Memory\ Bandwidth
\]

rather than Tensor Core throughput.

---

# 14. KV cache fragmentation

Now we have a runtime problem.

Imagine:

```text
Request A → 37,000 tokens
Request B → 4,500 tokens
Request C → 19,000 tokens
Request D → finishes
Request E → 12,000 tokens
```

If KV is allocated as one contiguous block:

```text
████████ A █████
██ B ██
████ C ███████
      FREE
████ E ███
```

you get fragmentation.

You may technically have enough free VRAM but not enough usable contiguous allocation.

---

# 15. Paged KV management problem

PagedAttention solves this by allocating KV in blocks:

```text
Physical KV memory

[Block 0] A
[Block 1] C
[Block 2] A
[Block 3] E
[Block 4] B
[Block 5] C
...
```

Logical sequence:

```text
Request A:
Block 0 → Block 2 → ...

Request C:
Block 1 → Block 5 → ...
```

This gives:

- less fragmentation
- dynamic allocation
- better batching
- easier sharing
- better GPU utilization

But:

> **Paged KV does not reduce the mathematical amount of KV required.**

It improves **memory management**, not the fundamental cache size.

---

# 16. KV cache eviction problem

Suppose context is:

```text
1,000,000 tokens
```

You cannot necessarily keep everything.

Then you need policies:

```text
Which KV entries should remain?
Which can be evicted?
Which can be compressed?
Which should be recomputed?
```

Possible strategies:

- sliding window
- attention sink
- token eviction
- KV compression
- hierarchical memory
- CPU/NVMe offload
- recurrent/SSM state

But every strategy risks losing information.

So:

\[
\boxed{
Memory\ reduction
\leftrightarrow
Information\ retention
}
\]

---

# 17. KV quantization problem

Instead of:

```text
FP16
2 bytes
```

use:

```text
FP8
1 byte
```

or potentially lower precision.

Approximately:

\[
KV_{FP8}\approx\frac{1}{2}KV_{FP16}
\]

Great for capacity and bandwidth.

But now:

\[
Precision\downarrow
\]

can cause:

- attention score error
- long-context degradation
- numerical instability
- layer/head-specific sensitivity

So again:

\[
Memory
\leftrightarrow
Quality
\]

---

# 18. Prefill vs Decode have different GPU problems

This is extremely important architecturally.

### Prefill

```text
Prompt
  ↓
many tokens simultaneously
  ↓
large matrix operations
  ↓
Tensor Cores
```

Often:

\[
Compute\ bound
\]

### Decode

```text
one new token
     ↓
Q
     ↓
read huge KV cache
     ↓
attention
     ↓
next token
```

Often:

\[
Memory\ bandwidth\ bound
\]

Therefore one optimization doesn't necessarily optimize both.

---

# 19. TTFT vs TPOT trade-off

Two important production metrics:

### TTFT

Time To First Token.

Mostly influenced by:

```text
Prefill
+
queueing
+
tokenization
```

### TPOT

Time Per Output Token.

Mostly influenced by:

```text
Decode
+
KV bandwidth
+
GPU memory traffic
```

So you can have:

```text
Excellent TTFT
Poor TPOT
```

or:

```text
Poor TTFT
Excellent TPOT
```

Architecture and runtime need to optimize both.

---

# 20. Prefix caching problem

If 1,000 users ask questions against the same system prompt:

```text
System prompt
      ↓
Same prefix
      ↓
User-specific suffix
```

recomputing the same prefix is wasteful.

Prefix KV caching can reuse it.

But now you need:

```text
KV cache identity
+
reference counting
+
eviction
+
memory sharing
+
security isolation
```

So cache reuse itself becomes a systems problem.

---

# 21. MLA addresses a deeper problem

GQA says:

> Share K/V heads.

MLA says:

> Don't cache the full K/V representation; cache a compressed latent representation.

Conceptually:

\[
h_t\rightarrow c_t
\]

where:

\[
c_t=W^{DKV}h_t
\]

and cache:

\[
c_t
\]

rather than full K/V.

This attacks the **representation dimension**, not simply the number of heads.

So the progression is roughly:

```text
MHA
 │
 │ reduce H_KV
 ↓
GQA
 │
 │ extreme H_KV sharing
 ↓
MQA
 │
 │ compress representation
 ↓
MLA
```

---

# 22. Cross-layer KV problem

Standard Transformer:

```text
Layer 1 → KV
Layer 2 → KV
Layer 3 → KV
...
Layer 32 → KV
```

KV memory contains the factor:

\[
L
\]

Cross-layer approaches ask:

> Do we really need completely independent KV states for every layer?

If KV can be shared across layers:

\[
L\downarrow
\]

and therefore:

\[
M_{KV}\downarrow
\]

This attacks a **different dimension** than GQA.

---

# 23. Hybrid SSM/Attention problem

Another architectural direction:

```text
75% SSM / linear attention
25% full attention
```

Most layers don't maintain conventional full KV caches.

This attacks:

\[
T\times H_{KV}\times D
\]

by replacing explicit token-level retrieval with a recurrent/compressed state.

But:

```text
SSM
 ↓
efficient state
 ↓
poor arbitrary token retrieval

Attention
 ↓
expensive KV
 ↓
excellent arbitrary retrieval
```

So the architecture has to decide:

> **Where do I need exact retrieval and where can I use compressed memory?**

---

# 24. The real architecture problem

All of these techniques are attacking different terms:

\[
\boxed{
M_{KV}=2LBT H_{KV}DS
}
\]

| Technique            | Attacks                           |
| -------------------- | --------------------------------- |
| MHA → GQA           | \(H_{KV}\)                        |
| GQA → MQA           | \(H_{KV}\)                        |
| KV quantization      | \(S\)                             |
| Sliding window       | \(T\)                             |
| MLA                  | effective\(H_{KV}D\)              |
| CLA                  | \(L\)                             |
| YOCO                 | effective layer KV duplication    |
| SSM                  | replaces token KV                 |
| Paged KV             | fragmentation/allocation          |
| Prefix caching       | redundant prefill                 |
| KV offload           | VRAM pressure                     |
| FlashAttention       | memory movement/workspace         |
| Continuous batching  | GPU utilization                   |
| Speculative decoding | reduces target-model decode steps |

This is the key architectural map.

---

# 25. GPU-side problem map

You can therefore classify the problems like this:

```text
                     GPU INFERENCE
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
     CAPACITY          BANDWIDTH           COMPUTE
       │                  │                  │
       ▼                  ▼                  ▼
   Model weights       KV reads          QKV GEMM
   KV cache            KV writes         FFN
   Activations         HBM traffic       Projection
   Workspace           Memory wall       Tensor cores
       │                  │                  │
       └────────────┬─────┴──────────────────┘
                    │
                    ▼
              PERFORMANCE
```

And KV management sits in the middle:

```text
                KV CACHE
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
   Capacity     Bandwidth   Allocation
       │           │           │
    VRAM        HBM BW     fragmentation
       │           │           │
       ▼           ▼           ▼
   context      TPOT       batching
   concurrency  latency    scheduling
```

---

# 26. The complete problem statement

For an AI Architect, I would summarize the challenge as:

> **How do we maximize useful tokens/sec/$ while maintaining model quality, given finite GPU memory capacity, finite memory bandwidth, finite compute throughput, and dynamically changing KV-cache requirements?**

Mathematically:

\[
\boxed{
\max
\frac{\text{Quality}\times\text{Throughput}}
{\text{Latency}\times\text{Cost}}
}
\]

subject to:

\[
Weights+KV+Activation+Workspace < VRAM
\]

and

\[
Latency_{decode}
\approx
\frac{KV\ bytes\ moved}{Memory\ Bandwidth}
\]

for memory-bound regimes.

---

# 27. The architecture decision tree

This is probably the most useful mental model for your AI Architect preparation:

```text
                  LONG-CONTEXT LLM
                         │
                         ▼
                KV cache too large?
                    /          \
                  YES           NO
                   │             │
                   ▼             ▼
                GQA/MQA      Keep MHA/GQA
                   │
                   ▼
             Still too large?
                /       \
              YES        NO
               │          │
               ▼          ▼
             MLA       Continue
               │
               ▼
        Still too large?
               │
        ┌──────┴──────┐
        ▼             ▼
   KV quantization  Sliding window
        │             │
        └──────┬──────┘
               ▼
        Still too large?
               │
               ▼
          CLA / YOCO
               │
               ▼
        Still too large?
               │
               ▼
       Hybrid SSM/Attention
```

Then independently:

```text
KV fits?
   │
   ▼
Decode slow?
   │
   ▼
Check arithmetic intensity
   │
   ├── Compute bound
   │       ↓
   │    Tensor cores
   │    quantization
   │    kernel optimization
   │
   └── Memory bound
           ↓
        GQA/MQA
        KV quantization
        Paged KV
        FlashAttention
        cache locality
        higher BW GPU
```

### The single most important insight

Don't think of **KV cache as merely a memory-storage problem**.

It is simultaneously:

\[
\boxed{
Capacity +
Bandwidth +
Latency +
Concurrency +
Scheduling +
Quality
}
\]

And that's why modern LLM architecture is increasingly about **controlling the information movement through the GPU memory hierarchy**, not merely increasing FLOPS.


---

# PART — KV cache as a virtual-memory problem: fragmentation, PagedAttention, FlashAttention, scheduling

Yes. The clean way to understand this is to treat **KV cache as an OS-style virtual-memory problem**, not just a tensor-memory problem.

The hard production problem is:

> **Requests arrive with unknown final lengths, KV grows token-by-token, requests finish at arbitrary times, and we want to pack as many active sequences as possible into finite GPU memory without fragmentation or expensive copying.**

PagedAttention is primarily a solution to **allocation/fragmentation**. FlashAttention is primarily a solution to **attention computation and HBM traffic**. They solve different layers of the problem and can be used together. [arXiv](https://arxiv.org/abs/2309.06180)

---

# 1. First: what exactly is fragmentation?

Suppose a GPU has 16 GB available for KV.

Imagine four requests:

```text
GPU KV memory

┌──────────────────────────────────────────────┐
│ Request A │ Request B │ Request C │ Request D│
└──────────────────────────────────────────────┘
```

But requests don't finish together.

```text
Time T0:

[A A A A][B B B][C C C C C][D D]
```

Now B finishes:

```text
[A A A A][FREE FREE FREE][C C C C C][D D]
```

Then a new request E arrives requiring 4 units:

```text
[A A A A][FREE FREE FREE][C C C C C][D D][E E E E]
```

There might be enough **total** free memory, but the free region may not be usable in the way the allocator requires.

This leads to two different concepts.

---

# 2. External fragmentation

External fragmentation means:

> **Free memory exists, but it is split into separated regions.**

Example:

```text
GPU memory

[A A A][FREE][B B B B][FREE][C C][FREE][D D D D]
          ↑              ↑
       2 GB             3 GB

Total free = 5 GB
```

Suppose the next request requires:

```text
4 GB contiguous
```

You have:

\[
3GB+2GB=5GB
\]

but no 4 GB contiguous region.

So allocation fails even though:

\[
FreeMemory > RequiredMemory
\]

This is classic **external fragmentation**.

---

# 3. Internal fragmentation

Internal fragmentation is different.

Suppose we allocate memory in fixed blocks:

```text
Block size = 16 tokens
```

Request A currently has:

```text
33 tokens
```

It needs:

```text
16 + 16 + 16 = 48 token slots
```

Therefore:

```text
Used:
33 tokens

Allocated:
48 token slots

Waste:
15 token slots
```

That's:

\[
InternalWaste=Allocated-Used
\]

So:

\[
15/48\approx31\%
\]

of the allocation is temporarily unused.

---

# 4. Why contiguous KV allocation is particularly bad

Imagine a traditional implementation saying:

> "When request starts, allocate enough memory for its maximum sequence length."

Suppose:

```text
max context = 32K
```

Request A may actually generate:

```text
500 tokens
```

but you reserved:

```text
32K
```

You have:

```text
Reserved
████████████████████████████████ 32K

Actually used
█ 500
```

That's massive internal waste.

But you don't know the final length.

This is the key problem.

---

# 5. The unknown final sequence length problem

At arrival:

```text
Request A
Prompt = 4K
Max output = ?
```

Maybe it generates 100 tokens, maybe 2K, maybe 20K, maybe it gets stopped at EOS. You don't know beforehand.

Therefore \(T_{final}\) is unknown. But KV grows:

\[
KV(T)\propto T
\]

So the allocator must support \(KV_1\rightarrow KV_2\rightarrow KV_3\rightarrow...\) dynamically.

---

# 6. Naive solution: contiguous dynamic allocation

```text
Start:
[A A A]

Need more:
[A A A][A A]

Need more:
[A A A][A A][A A A]
```

But eventually you might need to move A:

```text
Before:

[A A A][B B B][C C C]

A grows:

[A A A][A A][B B B][C C C]
```

If there isn't enough space after A, you may need:

```text
copy A → another location
```

That is expensive. And copying a large KV cache during generation is exactly what you don't want.

---

# 7. PagedAttention's key idea

PagedAttention says:

> Don't allocate KV as one giant contiguous tensor per request.

Instead:

\[
KV = \text{sequence of fixed-size blocks}
\]

For example, block size = 16 tokens. Request A:

```text
Logical sequence:

A A A A A A A A A A A A A A A A
A A A A A A A A A A A A A A A A
A A A A A

       ↓

Physical blocks:

Block 7
Block 19
Block 4
```

They don't have to be physically contiguous.

---

# 8. Virtual address vs physical block

This is exactly the OS analogy.

```text
CPU virtual memory:          Paged KV:

Virtual pages                Logical token positions
      ↓                            ↓
Page table                   KV block table
      ↓                            ↓
Physical RAM                 Physical GPU KV blocks
```

For example:

```text
Request A logical KV:

Token 0-15   → Block 7
Token 16-31  → Block 19
Token 32-47  → Block 4
Token 48-63  → Block 22
```

The sequence is logically contiguous. Physical memory doesn't need to be.

---

# 9. Now the unknown-length problem becomes easy

```text
Start:            allocate Block 7
After 16 tokens:  need more → Block 19
Need more:        Block 4
Need more:        Block 22
```

No need to move the previous blocks. Physically the blocks can be anywhere.

This is the fundamental reason PagedAttention is powerful. The original vLLM/PagedAttention work specifically identifies dynamically growing/shrinking KV caches and fragmentation as major obstacles to batching, and uses paging to achieve near-zero KV waste while enabling sharing. [arXiv](https://arxiv.org/abs/2309.06180)

---

# 10. What happens when a request finishes?

```text
GPU block pool

Block 0 → A
Block 1 → C
Block 2 → B
Block 3 → A
Block 4 → D
Block 5 → C
Block 6 → A
Block 7 → B
```

Request B finishes. Its blocks (2, 7) are returned to the free pool:

```text
Block 0 → A
Block 1 → C
Block 2 → FREE
Block 3 → A
Block 4 → D
Block 5 → C
Block 6 → A
Block 7 → FREE
```

Now the next request E arrives. It can immediately consume Block 2 and Block 7. No need to create a contiguous region.

---

# 11. This eliminates most external fragmentation

```text
Traditional:  [A][FREE][B][FREE][C][FREE]

Paged block pool:  [ A ][FREE][ B ][FREE][ C ][FREE]
```

The allocator doesn't care that free blocks are scattered. A new request simply gets FREE → Block 2, FREE → Block 7, FREE → Block 11, ...

\[
\boxed{Physical\ contiguity\ is\ no\ longer\ required}
\]

---

# 12. But PagedAttention doesn't eliminate internal fragmentation

Block size = 16 tokens, request currently has 33 tokens:

```text
Block 1 = tokens 0-15
Block 2 = tokens 16-31
Block 3 = token 32       ← 15 unused token slots
```

So there is still internal fragmentation, but it is bounded. With block size \(P\):

\[
Waste < P
\]

per active sequence's final block.

```text
Huge max-length reservation  →  BAD
Fixed small blocks           →  bounded waste
```

That's what "near-zero waste" means in the PagedAttention context: not literally zero bytes wasted, but dramatically reducing the waste caused by reserving large contiguous regions.

---

# 13. Block size becomes an engineering choice

- **Block = 4 tokens:** very little internal waste, but more blocks, larger block tables, more bookkeeping, potentially more indirection.
- **Block = 256 tokens:** fewer blocks, less metadata, but more internal waste.

\[
\boxed{BlockSize \leftrightarrow Fragmentation \leftrightarrow Metadata \leftrightarrow Kernel\ efficiency}
\]

This is an allocator/kernel co-design problem.

---

# 14. Now where does FlashAttention enter?

- **PagedAttention** solves: **Where do I store KV?**
- **FlashAttention** solves: **How do I efficiently compute attention while minimizing HBM ↔ SRAM traffic?**

They operate at different levels.

---

# 15. Standard attention memory problem

For \(S=QK^T\) you have \(S\in\mathbb{R}^{T\times T}\). For T = 100K, T × T = 10^10. You don't want to materialize that giant attention matrix.

FlashAttention tiles the computation: it loads Q and K/V tiles into faster on-chip memory, computes partial attention (online softmax), and avoids writing the full \(T\times T\) matrix to HBM. The original FlashAttention paper frames this as an IO-aware exact-attention algorithm that reduces HBM reads/writes through tiling. [arXiv](https://arxiv.org/abs/2205.14135)

---

# 16. FlashAttention does NOT solve KV capacity

A 10 GB KV cache stays 10 GB; FlashAttention just reads pieces of it efficiently.

\[
\boxed{FlashAttention \neq KV\ compression} \qquad \boxed{FlashAttention \neq KV\ allocator}
\]

---

# 17. PagedAttention + FlashAttention

```text
Request → Logical KV sequence → Block table → scattered GPU blocks
        → Attention kernel (FlashAttention-style, gathers via block table) → Output
```

PagedAttention tells the system: *KV blocks are here.* The attention kernel says: *I'll process those blocks efficiently.*

---

# 18. The request lifecycle

```text
Request arrives
      │
      ▼
Tokenize
      │
      ▼
Check prefix cache
      │
      ├──── HIT ────► reuse KV blocks
      │
      └──── MISS ───► allocate new blocks
                         │
                         ▼
                       Prefill
                         │
                         ▼
                     Decode loop
                         │
              ┌──────────┴──────────┐
              │                     │
          generate token        EOS/stop
              │                     │
              ▼                     ▼
        allocate block          free blocks
        when needed
```

---

# 19. Decode allocation

Block size 16, prompt = 100 tokens → \(\lceil100/16\rceil=7\) blocks (112 slots). Token #101–#112 still fit in block 7. At token #113 a new block is allocated; at token #129 another. Memory grows **on demand** — exactly what you want when the final length is unknown.

---

# 20. Request finishes early

Expected maximum = 32K, actual = 1.2K. A naive contiguous allocator might have reserved the full 32K. Paged allocation has only allocated \(\lceil1200/P\rceil\) blocks (75 at P=16). At EOS: blocks → refcount 0 → free pool, immediately reusable. Short requests don't monopolize memory.

---

# 21. Request does NOT finish: it keeps consuming blocks

GPU KV capacity = 10 GB; A = 3 GB, B = 2 GB, C = 4 GB; free = 1 GB. C needs +500 MB (fine), A needs +700 MB (fine), B needs +1 GB (not enough). The scheduler must decide:

1. don't admit another request
2. pause a request
3. evict KV
4. swap KV to CPU
5. recompute later
6. reduce batch
7. terminate/cancel
8. use KV compression

This is where **KV cache management becomes scheduling**.

---

# 22. Admission control

A good serving system shouldn't say "Free VRAM > 0 → accept request." It must estimate \(KV_{required}\) and whether the request can be sustained. E.g. budget 20 GB, used 15 GB (A 5, B 4, C 6), new request D with an 8K prompt and unknown generation: the scheduler can't blindly reserve the theoretical maximum; it needs dynamic allocation and a token/block budget.

---

# 23. Continuous batching

Static batching runs A B C D and waits until ALL finish (A: 20 tokens, B: 500, C: 50, D: 1000 → GPU waits for D). Continuous (iteration-level) batching replaces finished requests every iteration:

```text
Iteration 1: A B C D   → A finishes, E enters
Iteration 2: B C D E   → C finishes, F enters
Iteration 3: B D E F
```

Because KV blocks are dynamically allocated/freed, this works naturally with a paged allocator.

---

# 24. Scheduling and memory become coupled

The scheduler must consider prompt length, output length, KV blocks, prefix sharing, GPU memory, prefill vs decode workload, SLO and priority:

\[
\boxed{Memory\text{-}aware\ scheduling}
\]

---

# 25–26. Prefix caching and reference counting

100 requests sharing a system prompt + RAG instructions + policy can share one set of prefix KV blocks (vLLM automatic prefix caching uses block hashes; SGLang RadixAttention uses a radix tree). Shared blocks need reference counting: refcount(Block 7) = 3 → A finishes → 2 → B → 1 → C → 0 → FREE. This is copy-on-write / shared-memory management.

---

# 27–31. Long-context options

| Option | Idea | Cost |
|---|---|---|
| Full KV | exact retrieval, simple semantics | huge memory + bandwidth |
| Sliding window | keep last W tokens, O(W) not O(T) | distant info discarded |
| KV compression | quantization, learned compression, latent KV | quality may drop |
| MLA | cache a compressed latent c instead of K,V | needs (re)training |
| Hybrid SSM + attention | most layers keep a recurrent state; few keep full KV | weaker arbitrary retrieval |
| Context parallelism | shard 1M tokens across GPUs (4 × 250K) | bottleneck moves from HBM to NVLink/InfiniBand |

---

# 32. FlashAttention's role in long context

FlashAttention avoids materializing \(T\times T\) scores; FlashAttention-2 improves parallelism and work partitioning (much higher A100 utilization). [arXiv](https://arxiv.org/abs/2307.08691) But long context has three separate problems:

```text
Long context
 ├── KV capacity problem   → GQA / MLA / quantization
 ├── KV bandwidth problem  → GQA / quantization / architecture
 └── Attention compute/IO  → FlashAttention
```

---

# 33. The complete SOTA stack

```text
                    LLM SERVING
 ┌───────────────────────┼────────────────────────┐
MODEL                   CACHE                   KERNEL
GQA                    Paged KV             FlashAttention
MQA                    Prefix cache          FlashInfer
MLA                    KV quant              Fused kernels
CLA                    Offload
SSM                    Eviction
Hybrid                 Block allocator
                │
             RUNTIME: continuous batching, chunked prefill, speculative decoding,
                      memory-aware scheduling, P/D disaggregation
                │
              GPU: VRAM capacity, HBM bandwidth, tensor cores, NVLink
```

---

# 34. Five different problems

| Problem | Question | Typical solution |
|---|---|---|
| **Capacity** | Does KV fit? | GQA, MLA, quantization |
| **Fragmentation** | Can free memory be reused? | Paged KV |
| **Bandwidth** | Can GPU read KV fast enough? | GQA, quantization, better GPU |
| **Attention compute/IO** | Can attention run efficiently? | FlashAttention |
| **Scheduling** | Which requests run now? | Continuous/memory-aware batching |

| Long-context problem | Solution |
|---|---|
| KV too large | GQA/MLA/KV quant |
| Need exact distant retrieval | Full attention |
| Mostly local context | Sliding window |
| Huge context | Hybrid attention/SSM |
| Prefix repetition | Prefix caching |
| >1 GPU context | Context parallelism |
| Prefill bottleneck | Chunked prefill / P-D disaggregation |
| Decode bottleneck | KV optimization / speculative decoding |

---

# 35. "Next request / unused memory / request finish" mental model

```text
                    GPU KV BLOCK POOL
        ┌────────────────────────────────────┐
        │ F │ A │ C │ F │ B │ A │ C │ F │ D │
        └────────────────────────────────────┘
```

- **New request arrives:** E arrives → scheduler checks capacity → allocate FREE blocks → E gets scattered physical blocks.
- **E generates more tokens:** no space in current block → allocate another FREE block → append mapping.
- **E finishes:** E blocks → refcount = 0 → return to FREE pool.
- **Next request:** F arrives → reuses those blocks.

No giant `malloc`. No need for contiguous KV. No need to predict exact final sequence length.

---

# 36. Free blocks aren't necessarily enough

1000 blocks; 900 allocated, 100 free. Request E needs 50 blocks now but could eventually need 500. The system cannot guarantee that → the scheduler deals with **uncertain future KV demand**: memory allocation + admission control + scheduling + eviction + fairness. That's why serving systems maintain block managers, schedulers, prefix trees, eviction policies and sometimes CPU offload.

---

# 37. The real optimization objective

\[
\boxed{\text{Maximum useful throughput subject to latency and quality SLOs}}
\]

Option A: 95% VRAM utilization but heavy fragmentation and poor TPOT. Option B: 85% VRAM utilization but excellent batching and low TPOT. Option B may give higher real throughput. \(VRAM\ utilization \neq GPU\ efficiency\).

---

# 38. Architecture choice framework (interview order)

1. Can the weights fit? \(W_{model}<VRAM\)
2. How much KV do I need? \(M_{KV}=2LBT H_{KV}DS\)
3. Is KV causing capacity pressure? → GQA → KV quantization → MLA → sliding window → hybrid
4. Is fragmentation causing poor batching? → Paged KV / block allocation
5. Is attention compute/IO the bottleneck? → FlashAttention / FlashInfer / fused kernels
6. Is decode memory-bandwidth bound? → GQA, KV quant, higher-bandwidth GPU, cache locality
7. Is scheduling the bottleneck? → continuous batching, memory-aware scheduling, chunked prefill, prefix-aware scheduling
8. Is context extremely long? → MLA, KV compression, sliding window, hybrid SSM, context parallelism
9. Is TTFT the problem? → prefix caching, chunked prefill, prefill optimization, P/D disaggregation
10. Is TPOT the problem? → KV bandwidth, GQA/MLA, KV quantization, FlashAttention/FlashInfer, speculative decoding

## The most important picture

```text
                    LONG-CONTEXT LLM
             ┌─────────────┴─────────────┐
        MODEL ARCHITECTURE          SERVING SYSTEM
       GQA   MLA   SSM           Paged KV  Prefix Cache  Scheduler
             ▼                            ▼
         KV SIZE                    KV LIFECYCLE
             └────────────┬───────────────┘
                          ▼
                       GPU VRAM  (Capacity · Bandwidth)
                          ▼
                   Attention Kernel (FlashAttention)
                          ▼
                       OUTPUT
```

> **GQA/MLA/SSM decide how much historical information must exist; PagedAttention decides how that information is allocated and reclaimed; FlashAttention decides how efficiently attention consumes it; and the scheduler decides which requests are allowed to occupy the GPU at each iteration.**

---

# PART — Virtual memory → KV pages → logical tokens → physical GPU blocks

Connect **virtual memory → fixed-size KV pages → logical token positions → physical GPU blocks**.

**Contiguous KV:** request A grows, final length unknown; outgrowing a contiguous region means allocate bigger region → copy existing KV → free old region. Terrible during inference.

**Paged:** KV block = 16 tokens. Logical pages (tokens 0–15, 16–31, 32–47, 48–63) map to physical GPU blocks 17, 93, 4, 51 — not adjacent, yet A still sees Page 0 → Page 1 → Page 2 → Page 3. Same abstraction as Virtual Page → Page Table → Physical RAM Page.

**Growth without copying:** prompt = 100 tokens → \(\lceil100/16\rceil=7\) blocks, A → [17, 93, 4, 51, 82, 12, 65]. Block 65 holds tokens 96–111; for tokens 112–127 the allocator takes FREE block 28 → A → [17, 93, 4, 51, 82, 12, 65, 28]. **Nothing is copied.**

> **KV cache grows by allocating another page, not by resizing a contiguous tensor.**

**Multiple requests:**

```text
GPU KV BLOCK POOL
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│ A  │ C  │FREE│ B  │ A  │FREE│ C  │ D  │FREE│ A  │
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘

A → [0,4,9]   B → [3]   C → [1,6]   D → [7]
E needs 3 blocks: FREE pool = [2,5,8] → E → [2,5,8]
After: [A][C][E][B][A][E][C][D][E][A]
```

**OS analogy table:**

| | OS virtual memory | LLM KV cache |
|---|---|---|
| Logical | Virtual address/page | Token position/page |
| Mapping | Page table | Block table |
| Physical | RAM page | GPU KV block |
| Reclaim | Page allocator | KV block allocator |

Interview answer: **"Paged KV caching applies virtual-memory principles to autoregressive inference. Logical token positions are mapped through a block table to non-contiguous physical GPU KV blocks, allowing KV to grow incrementally and allowing freed blocks to be immediately reused."**

**Future demand:** 1000 blocks; A 300, B 200, C 300 → 800 used, 200 free. D needs 30 now, but final output could be 50, 500 or 10,000 tokens → UNKNOWN FUTURE KV → memory scheduler → admit / pause / evict-offload. The problem evolves from "how do I allocate memory?" to **"how do I schedule uncertain future memory demand?"**

**Three layers:**
- **GQA / MQA / MLA** → how much KV do I need to store?
- **Paged KV / block allocator** → how do I allocate, grow, share and reclaim KV?
- **FlashAttention / FlashInfer** → how efficiently do I compute attention using that KV?

> **Virtual memory solves the physical allocation problem; PagedAttention applies that idea to KV cache; continuous batching solves request turnover; and FlashAttention solves efficient attention computation.** The remaining hard problem is **memory-aware scheduling under unknown future sequence length**.

---

# PART — External vs internal fragmentation vs duplication

**External fragmentation = free memory exists, but is non-contiguous.**

```text
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│  A   │ FREE │  B   │ FREE │  C   │ FREE │  D   │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

Paged: token 0–15 → block 7, 16–31 → block 2, 32–47 → block 11, 48–63 → block 4. The KV cache is **logically contiguous but physically non-contiguous**: \(\text{Logical continuity} \neq \text{physical continuity}\). Scattered GPU memory becomes useful.

**Internal fragmentation:** block size 16, sequence 18 tokens → 2 blocks = 32 slots, 18 used, **14 wasted**, all in the final partial block. Generally \(Waste < BlockSize\).

| Problem | Contiguous KV | Paged KV |
|---|---|---|
| External fragmentation | High | **Greatly reduced** |
| Physical contiguous memory required | Yes | **No** |
| Internal fragmentation | Possible (huge with max-len reservation) | **Still exists (bounded)** |
| Final partially filled block | — | Yes |
| Memory allocation | Large contiguous region | Fixed-size blocks |
| Growing sequence | Difficult/expensive | Allocate another block |
| Scattered free blocks | Often difficult to use | **Can be reused** |

Don't say "PagedAttention eliminates fragmentation." Say: **"PagedAttention substantially reduces external fragmentation by allowing KV cache to occupy non-contiguous physical GPU blocks, while introducing bounded internal fragmentation from partially filled blocks."**

**Duplication (a different issue):** 100 requests with the same system prompt store \(100\times KV_{prefix}\) without sharing. With prefix sharing one set of blocks is referenced by all; refcount(Block 7) = 3 → 2 → 1 → 0 → FREE.

```text
                 KV MEMORY PROBLEMS
       ┌─────────────────┼──────────────────┐
 External           Internal            Duplication
 fragmentation      fragmentation       / redundancy
 scattered          partially           same KV stored
 free memory        filled block        multiple times
       ↓                 ↓                  ↓
 Paged KV           smaller block       Prefix caching
 / block allocator  size                / sharing

 + Unknown final sequence length → KV grows dynamically → dynamic block allocation → Paged KV
```

**Interview-ready sentence:** **Contiguous KV allocation suffers from external fragmentation because growing requests require contiguous physical memory. PagedAttention breaks KV into fixed-size blocks, so logical token sequences can map to scattered physical GPU blocks, making fragmented free memory reusable. The trade-off is bounded internal fragmentation in the partially filled final block. Prefix caching then addresses a separate problem—duplicated KV across requests sharing the same prefix.**


# PART — MHA vs GQA vs MQA vs MLA by "what is cached per token"

Worked example: H = 32, d_h = 128, d_model = H·d_h = 4096 (numbers per token per layer).

| Architecture | Q heads | KV heads | What is cached | KV / token / layer | Example |
|---|---:|---:|---|---|---:|
| MHA | H | H | K + V for every head | 2·H·d_h = 2·d_model | 8192 |
| GQA-8 | H | G = 8 | K + V for fewer heads (H/G = 4 Q heads per KV head) | 2·G·d_h | 2048 (4×) |
| MQA | H | 1 | one shared K, V | 2·d_h | 256 (32×) |
| MLA | H | — | compressed latent c^KV (d_c = 512) + decoupled RoPE key k^R (d_R = 64) | ≈ d_c + d_R | 576 (14.2×) |

- **GQA/MQA reduce the number of KV heads; MLA compresses the KV representation itself.** h_t (4096) → W^DKV → c^KV (512, cached); W^UK / W^UV up-project to 32 per-head K, V only when needed (and are absorbed into the query/output projections at decode). K and V come from the same latent, so the factor 2 disappears.
- MLA caches more than MQA here (576 vs 256) but keeps a distinct K/V view per head; its budget equals GQA with 2.25 KV heads. (DeepSeek-V2 has 128 heads, so its own ratio is 32,768 / 576 ≈ 57×.)
- Bandwidth ∝ KV size: 4 chats × 8K tokens, L = 32, BF16 on an A100-40GB (1.555 TB/s) → per decode step MHA reads 16 GiB (~11 ms), GQA-8 4 GiB (~2.8 ms), MLA 1.125 GiB (~0.8 ms), MQA 512 MiB (~0.35 ms).
- System picture: **MHA/GQA/MLA decide the representation and amount of KV; PagedAttention decides how it is physically allocated; block sharing removes duplicate KV; FlashAttention decides how efficiently the GPU consumes it.**

Formulas: KV_MHA = 2·H·d_h, KV_GQA = 2·G·d_h, KV_MQA = 2·d_h, KV_MLA ≈ d_c + d_R.
Diagrams: `comparison.excalidraw` sections 6–11; real-model numbers in `sota.excalidraw`.
