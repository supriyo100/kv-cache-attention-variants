# Serving stack & scheduler

[Session 10](../sessions/10_pagedattention_vllm.md) introduces PagedAttention. A production
serving engine wraps it in two more layers: a **stack** that separates *where* KV lives from
*how* attention is computed, and a **scheduler** that treats KV memory as the resource every
request competes for.

## The serving stack

PagedAttention decides **where** the KV cache is stored (blocks and block tables).
FlashAttention-style kernels decide **how** attention is computed over it (tiles in SRAM).
They solve different problems and work together. The drawing then builds up to the full stack,
context parallelism for million-token prompts, and a 10-step framework for choosing an
architecture.

![Serving stack: PagedAttention vs FlashAttention vs the whole SOTA picture](../assets/excalidraw/serving_stack.svg){ .excalidraw }

## The scheduler

Once memory is paged, serving becomes a scheduling problem: admit, pause or preempt requests
every iteration so the KV pool stays full of *useful* tokens, while meeting latency targets
(TTFT and TPOT).

![Serving scheduler: KV memory as a scheduling problem](../assets/excalidraw/serving_scheduler.svg){ .excalidraw }
