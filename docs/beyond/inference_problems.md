# Inference problems

Before choosing an optimization, name the problem. Serving an LLM runs into four kinds of limit,
and every technique in the course relieves one of them:

| Problem | The question | Relieved by |
|---|---|---|
| **Capacity** | Does the KV cache fit in GPU memory? | GQA / MQA / MLA, KV quantisation, eviction, offload |
| **Bandwidth** | Can HBM feed the cores fast enough during decode? | Smaller caches, FlashAttention / FlashDecoding, speculative decoding |
| **Allocation** | Is the memory we have actually holding tokens? | PagedAttention, continuous batching, memory-aware scheduling |
| **Quality** | Did the optimization cost accuracy? | Choosing lossless options first; validating lossy ones |

The [roofline drawing in session 08](../sessions/08_flashattention.md) explains why decode is
bandwidth-bound, and the [VRAM budget in session 02](../sessions/02_kv_cache_memory_math.md)
shows the capacity side with real numbers.

![LLM inference problems: capacity, bandwidth, allocation, quality](../assets/excalidraw/inference_problems.svg){ .excalidraw }
