"""inference_lab: small, tested reference models of LLM inference systems.

Original code for the Inference Engineering notebook site. Each module is a
readable model of one mechanism (KV memory math, paged allocation, prefix
sharing, scheduling, IO-aware attention, MLA absorption, KV quantization,
eviction, speculative decoding), not a production implementation. Where a
module follows a published system, its docstring names the paper and the
official repository.
"""

__all__ = [
    "disagg",
    "eviction",
    "flash",
    "fragmentation",
    "kv_math",
    "mla",
    "paged",
    "prefix",
    "quant",
    "scheduler",
    "specdec",
]
