# KV cache SOTA map

The course explains the core ideas. This page maps **every major technique in use today** onto
the term of the memory formula it attacks, or onto the cost of storing, recomputing or reading
the cache.

| Family | Attacks | Examples |
|---|---|---|
| ① Shrink per-token size | $H_{kv}$, $d_h$, $L$ (architecture) | GQA, MQA, MLA, cross-layer sharing (CLA, YOCO), TransMLA retrofit |
| ② Fewer bytes per element | $S$ | FP8 KV, INT4/INT2 (KIVI, KVQuant), NVFP4 |
| ③ Keep fewer tokens | $T$ | Sliding window, local:global layers, attention sinks, H2O, SnapKV |
| ④ Manage memory better | fragmentation, placement | PagedAttention, continuous batching, chunked prefill, offload, disaggregation |
| ⑤ Don't recompute | prefill / TTFT | Prefix caching, RadixAttention, provider prompt caching, KV-aware routing |
| ⑥ Read it faster | HBM traffic per step | FlashAttention, FlashDecoding, GQA/MLA-aware kernels, speculative decoding |

## The map, technique by technique

The top of the drawing is the overview. Below it, each family gets a row of cards with the worked
numbers for one model (Llama-3-8B shape: $L = 32$, $H_Q = 32$, $H_{kv} = H_Q/4 = 8$,
$d_h = 128$), and the last chart shows how the techniques stack on a fixed model.

![KV cache optimization: SOTA map with a worked example for every technique](../assets/excalidraw/kv_cache_sota_map.svg){ .excalidraw }

## What frontier models and engines actually do

What today's open models cache per token, the toolbox grouped by problem, a 32K-chat capacity
example, a reference serving stack, and which tool to reach for first for each symptom.

![SOTA: how today's models and serving engines handle the KV cache](../assets/excalidraw/sota.svg){ .excalidraw }
