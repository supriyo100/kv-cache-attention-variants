---
title: References
description: Every paper cited on this site, with titles, authors and dates fetched from arXiv and checked automatically.
---

# References

<p class="lede">Every paper cited on this site. Titles, authors and dates for arXiv papers are fetched from the arXiv API by <code>tools/references.py</code>, which fails the build step if a cited ID does not match its paper. The page is generated; do not edit it by hand.</p>

## Foundations

| Paper | Authors | Year | Cited in step |
|---|---|---|---|
| [A Proof for the Queuing Formula: L = λW](https://doi.org/10.1287/opre.9.3.383) (Operations Research 9(3), 1961) | Little | — | 1 |
| [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785) (Communications of the ACM 52(4), 2009) | Williams, Waterman, Patterson | — | 2 |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Ashish Vaswani et al. | 2017 | 3 |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | Jared Kaplan et al. | 2020 | 2 |
| [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) | Reiner Pope et al. | 2022 | 4 |

## Architecture

| Paper | Authors | Year | Cited in step |
|---|---|---|---|
| [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) | Noam Shazeer | 2019 | 3, 5 |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | Jianlin Su et al. | 2021 | 5 |
| [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) | Joshua Ainslie et al. | 2023 | 3, 5 |
| [Mistral 7B](https://arxiv.org/abs/2310.06825) | Albert Q. Jiang et al. | 2023 | 11 |
| [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) | DeepSeek-AI et al. | 2024 | 3, 5 |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | DeepSeek-AI et al. | 2024 | 5, SOTA |
| [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887) | Opher Lieber et al. | 2024 | 11 |
| [Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](https://arxiv.org/abs/2502.14837) | Tao Ji et al. | 2025 | 5 |
| [TransMLA: Multi-Head Latent Attention Is All You Need](https://arxiv.org/abs/2502.07864) | Fanxu Meng et al. | 2025 | 5 |
| [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) | Gemma Team et al. | 2025 | 11 |

## Memory and sharing

| Paper | Authors | Year | Cited in step |
|---|---|---|---|
| [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | Woosuk Kwon et al. | 2023 | 6, 7, 8, 9 |
| [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) | Lianmin Zheng et al. | 2023 | 8, SOTA |
| [vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention](https://arxiv.org/abs/2405.04437) | Ramya Prabhu et al. | 2024 | 7 |

## Scheduling and serving

| Paper | Authors | Year | Cited in step |
|---|---|---|---|
| [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) (OSDI 2022) | Yu et al. | — | 9 |
| [Splitwise: Efficient generative LLM inference using phase splitting](https://arxiv.org/abs/2311.18677) | Pratyush Patel et al. | 2023 | 12 |
| [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310) | Amey Agrawal et al. | 2024 | 9 |
| [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670) | Yinmin Zhong et al. | 2024 | 1, 12 |
| [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) | Ruoyu Qin et al. | 2024 | 12, SOTA |

## Kernels

| Paper | Authors | Year | Cited in step |
|---|---|---|---|
| [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) | Maxim Milakov and Natalia Gimelshein | 2018 | 10 |
| [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) | Tri Dao et al. | 2022 | 10 |
| [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) | Tri Dao | 2023 | 10 |
| [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) | Jay Shah et al. | 2024 | 10 |
| [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005) | Zihao Ye et al. | 2025 | 7, 10 |

## Long context

| Paper | Authors | Year | Cited in step |
|---|---|---|---|
| [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) | Guangxuan Xiao et al. | 2023 | 11 |
| [H$_2$O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048) | Zhenyu Zhang et al. | 2023 | 11 |
| [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) | Zirui Liu et al. | 2024 | 11 |
| [SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469) | Yuhong Li et al. | 2024 | 11 |
| [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) | Amir Zandieh et al. | 2025 | 11, SOTA |

## Quantization

| Paper | Authors | Year | Cited in step |
|---|---|---|---|
| [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) | Elias Frantar et al. | 2022 | 12 |
| [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) | Guangxuan Xiao et al. | 2022 | 12 |
| [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) | Paulius Micikevicius et al. | 2022 | 12 |
| [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) | Ji Lin et al. | 2023 | 12 |

## Speculative decoding

| Paper | Authors | Year | Cited in step |
|---|---|---|---|
| [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) | Yaniv Leviathan et al. | 2022 | 12 |
| [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) | Charlie Chen et al. | 2023 | 12 |
| [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) | Tianle Cai et al. | 2024 | 12 |
| [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) | Yuhui Li et al. | 2024 | 12, SOTA |
| [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) | Yuhui Li et al. | 2024 | 12 |
| [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) | Yuhui Li et al. | 2025 | 12, SOTA |
| [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036) | Jian Chen et al. | 2026 | 12, SOTA |
| [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147) | Xin Cheng et al. | 2026 | 12, SOTA |

## Distributed

| Paper | Authors | Year | Cited in step |
|---|---|---|---|
| [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) | Mohammad Shoeybi et al. | 2019 | 12 |
| [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) | Hao Liu et al. | 2023 | 12 |
