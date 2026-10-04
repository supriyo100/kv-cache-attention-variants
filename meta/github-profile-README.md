<!--
Draft for the GitHub profile README. To use it: create a public repository named exactly
"supriyo100" (same as the username), add this file as README.md, then fill in the LinkedIn URL
below. Everything stated here is checkable from public repositories; edit anything that is out of date.
-->

# Supriyo Chakraborty

I work on how large language models are served: where the KV cache lives, why decode waits on
memory, and what techniques like paged attention, prefix caching and speculative decoding
trade off. I write it up as a public engineering notebook with derivations and tested code.

**Inference Engineering notebook:** https://supriyo100.github.io/kv-cache-attention-variants/
· LinkedIn: <!-- add your profile URL -->

### Inference engineering

- **[kv-cache-attention-variants](https://github.com/supriyo100/kv-cache-attention-variants)**:
  a 12-step path through LLM inference (prefill/decode, KV cache, MHA/GQA/MLA, paged KV,
  prefix sharing, scheduling, FlashAttention, long context, scaling out), with `inference_lab`,
  a tested PyTorch package of small reference implementations, and notebooks rebuilding
  current techniques (MLA absorption, RadixAttention, TurboQuant/KIVI, SnapKV, confidence-scheduled
  speculative decoding).

### Other work

- **Deep learning for geophysics**: seismic inversion with frequency-domain multi-scale U-Nets
  ([Orthoseisnet](https://github.com/supriyo100/Orthoseisnet), [UnetFFT](https://github.com/supriyo100/UnetFFT),
  [Seismic_inversion_thesis](https://github.com/supriyo100/Seismic_inversion_thesis)).
- **Optimization**: mixed-integer programming for production scheduling and inventory
  rebalancing ([Bicon_Manufacturing_MILP_optimization](https://github.com/supriyo100/Bicon_Manufacturing_MILP_optimization),
  [stada_inventory_rebalacing](https://github.com/supriyo100/stada_inventory_rebalacing)).
- **Agentic AI**: modular agent pipelines for automated research reports
  ([automated-research-report-generation](https://github.com/supriyo100/automated-research-report-generation),
  [AgenticAI](https://github.com/supriyo100/AgenticAI)).
