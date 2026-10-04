"""Prefill/decode disaggregation: when is moving the KV cache worth it?

Disaggregated serving (DistServe, Zhong et al., OSDI 2024, arXiv:2401.09670;
Splitwise; Mooncake, github.com/kvcache-ai/Mooncake; NVIDIA Dynamo) runs
prefill and decode on separate GPU pools. Prefill is compute-bound and sets
TTFT; decode is bandwidth-bound and sets ITL. Co-locating them means a long
prefill stalls every decode in the batch (or, with chunked prefill, inflates
each decode iteration). Separating them removes the interference, but the
prompt's KV must cross the network once:

    t_transfer = prompt_tokens * kv_bytes_per_token / link_bandwidth

Transfer is worth it when it is small next to the prefill itself, which holds
when the arithmetic intensity of prefill is high (long prompts, big models)
and the link is fast (NVLink, InfiniBand with RDMA).
"""

from __future__ import annotations

from . import kv_math

LINKS_GBS = {  # unidirectional, per GPU, nominal
    "NVLink 4 (H100)": 450.0,
    "InfiniBand NDR 400G": 50.0,
    "RoCE 200G": 25.0,
    "PCIe Gen5 x16": 64.0,
    "100GbE TCP": 12.5,
}


def transfer_time(m: kv_math.ModelShape, prompt: int, link_gbs: float, dtype: str = "bf16") -> float:
    return prompt * kv_math.kv_bytes_per_token(m, dtype) / (link_gbs * 1e9)


def prefill_time(m: kv_math.ModelShape, gpu: kv_math.GPU, prompt: int) -> float:
    flops, bytes_ = kv_math.step_cost(m, 1, prompt, prompt // 2)
    return kv_math.roofline_time(flops, bytes_, gpu)[0]


def transfer_ratio(m: kv_math.ModelShape, gpu: kv_math.GPU, prompt: int, link_gbs: float) -> float:
    """t_transfer / t_prefill. Below ~0.1 the transfer hides easily (layer-wise overlap)."""
    return transfer_time(m, prompt, link_gbs) / prefill_time(m, gpu, prompt)
