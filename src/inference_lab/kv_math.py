"""KV cache memory and prefill/decode cost models.

Per token, per layer, a standard attention layer caches one key and one value
vector for every KV head:

    bytes_per_token = L * 2 * H_kv * d_h * s

MLA (DeepSeek-V2/V3) caches one compressed latent c_KV of width d_c plus one
shared RoPE key of width d_R, with no separate K and V:

    bytes_per_token = L * (d_c + d_R) * s

Total cache for a batch: bytes_per_token * sum of sequence lengths.

The roofline helpers use the standard first-order model: a step takes
max(FLOPs / peak_flops, bytes / peak_bandwidth). Hardware numbers are vendor
datasheet peaks (dense, no sparsity) and are upper bounds, not measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DTYPE_BYTES = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5}


@dataclass(frozen=True)
class ModelShape:
    """Shape of a decoder-only transformer, as far as inference cost is concerned."""

    name: str
    layers: int
    d_model: int
    q_heads: int
    kv_heads: int
    head_dim: int
    params: float  # total parameters
    active_params: float | None = None  # per-token active parameters (MoE); None = dense
    attention: str = "gqa"  # "mha" | "mqa" | "gqa" | "mla"
    mla_latent: int = 0  # d_c for MLA
    mla_rope: int = 0  # d_R for MLA
    notes: str = field(default="", compare=False)

    @property
    def active(self) -> float:
        return self.active_params if self.active_params is not None else self.params


# Public configurations (from each model's config.json / technical report).
LLAMA3_8B = ModelShape("Llama-3-8B", 32, 4096, 32, 8, 128, 8.0e9, attention="gqa")
LLAMA3_70B = ModelShape("Llama-3-70B", 80, 8192, 64, 8, 128, 70.6e9, attention="gqa")
LLAMA2_7B = ModelShape("Llama-2-7B", 32, 4096, 32, 32, 128, 6.7e9, attention="mha")
DEEPSEEK_V3 = ModelShape(
    "DeepSeek-V3", 61, 7168, 128, 128, 128, 671e9, active_params=37e9,
    attention="mla", mla_latent=512, mla_rope=64,
    notes="MLA: caches c_KV (512) + shared RoPE key (64) per layer",
)
MODELS = {m.name: m for m in (LLAMA2_7B, LLAMA3_8B, LLAMA3_70B, DEEPSEEK_V3)}


@dataclass(frozen=True)
class GPU:
    name: str
    hbm_gb: float
    bandwidth_tbs: float  # TB/s
    bf16_tflops: float  # dense
    fp8_tflops: float  # dense (0 if unsupported)

    @property
    def ridge_bf16(self) -> float:
        """Arithmetic intensity (FLOP/byte) at which compute and bandwidth limits meet."""
        return self.bf16_tflops * 1e12 / (self.bandwidth_tbs * 1e12)


# NVIDIA datasheet peaks (dense).
A100_80G = GPU("A100 80GB SXM", 80, 2.039, 312, 0)
H100_SXM = GPU("H100 SXM", 80, 3.35, 989, 1979)
H200_SXM = GPU("H200 SXM", 141, 4.8, 989, 1979)
B200 = GPU("B200", 180, 8.0, 2250, 4500)
GPUS = {g.name: g for g in (A100_80G, H100_SXM, H200_SXM, B200)}


def kv_bytes_per_token(m: ModelShape, dtype: str = "bf16") -> float:
    """Bytes of KV cache one token occupies across all layers."""
    s = DTYPE_BYTES[dtype]
    if m.attention == "mla":
        return m.layers * (m.mla_latent + m.mla_rope) * s
    return m.layers * 2 * m.kv_heads * m.head_dim * s


def kv_bytes_per_token_variant(m: ModelShape, variant: str, dtype: str = "bf16",
                               groups: int | None = None) -> float:
    """KV bytes/token if model ``m`` used another attention variant (same L, H, d_h).

    variant: "mha" (H_kv = H_q), "mqa" (H_kv = 1), "gqa" (H_kv = groups), or
    "mla" (d_c = 4*d_h, d_R = d_h/2, the DeepSeek-V2 ratios).
    """
    s = DTYPE_BYTES[dtype]
    if variant == "mha":
        h = m.q_heads
    elif variant == "mqa":
        h = 1
    elif variant == "gqa":
        h = groups if groups is not None else m.kv_heads
    elif variant == "mla":
        return m.layers * (4 * m.head_dim + m.head_dim // 2) * s
    else:
        raise ValueError(variant)
    return m.layers * 2 * h * m.head_dim * s


def kv_cache_bytes(m: ModelShape, total_tokens: int, dtype: str = "bf16") -> float:
    return kv_bytes_per_token(m, dtype) * total_tokens


def weight_bytes(m: ModelShape, dtype: str = "bf16") -> float:
    return m.params * DTYPE_BYTES[dtype]


def max_concurrent_tokens(m: ModelShape, gpu: GPU, n_gpus: int = 1, weight_dtype: str = "bf16",
                          kv_dtype: str = "bf16", reserve_frac: float = 0.1) -> int:
    """Tokens of KV that fit after weights and a runtime reserve (activations, workspace)."""
    hbm = gpu.hbm_gb * 1e9 * n_gpus
    free = hbm * (1 - reserve_frac) - weight_bytes(m, weight_dtype)
    return max(0, int(free // kv_bytes_per_token(m, kv_dtype)))


# --- Prefill / decode cost --------------------------------------------------------------


def linear_flops(m: ModelShape, tokens: int) -> float:
    """Matmul FLOPs for ``tokens`` tokens through the weights: 2 * active_params per token."""
    return 2.0 * m.active * tokens


def attention_flops(m: ModelShape, new_tokens: int, context: int) -> float:
    """QK^T and PV FLOPs for ``new_tokens`` queries attending over ``context`` keys.

    Per layer per query head: 2*d_h*context (scores) + 2*d_h*context (weighted sum).
    Causal prefill averages context/2; callers pass the effective context.
    """
    return 4.0 * m.layers * m.q_heads * m.head_dim * new_tokens * context


def step_cost(m: ModelShape, batch: int, new_tokens: int, context: int,
              weight_dtype: str = "bf16", kv_dtype: str = "bf16") -> tuple[float, float]:
    """(FLOPs, HBM bytes) for one forward step.

    batch sequences, each adding ``new_tokens`` and attending over ``context``
    cached tokens. Bytes = weights read once + KV read for every sequence + new
    KV written. Activations are ignored (small next to weights and KV).
    """
    tokens = batch * new_tokens
    flops = linear_flops(m, tokens) + attention_flops(m, new_tokens, context) * batch
    kv = kv_bytes_per_token(m, kv_dtype)
    bytes_ = weight_bytes_active(m, weight_dtype) + batch * context * kv + tokens * kv
    return flops, bytes_


def weight_bytes_active(m: ModelShape, dtype: str = "bf16") -> float:
    """Weight bytes a step must stream. For MoE, approximated by active params."""
    return m.active * DTYPE_BYTES[dtype]


def roofline_time(flops: float, bytes_: float, gpu: GPU, dtype: str = "bf16") -> tuple[float, str]:
    peak = (gpu.fp8_tflops if dtype == "fp8" else gpu.bf16_tflops) * 1e12
    t_c = flops / peak
    t_m = bytes_ / (gpu.bandwidth_tbs * 1e12)
    return (t_c, "compute") if t_c >= t_m else (t_m, "memory")


def arithmetic_intensity(flops: float, bytes_: float) -> float:
    return flops / bytes_
