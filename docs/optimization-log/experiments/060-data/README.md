# Experiment 060 evidence

This directory deliberately separates evidence with different repetition contracts:

- `core/`: GPU visibility `1`; context 8/128/512, batch 1, two models, three cases,
  two frameworks, three fresh processes; one warm-up and two measured iterations.
- `batch/`: GPU visibility `3`; context 32, batch 1/2/4/8, one fresh process per row;
  one warm-up and two measured iterations. This is an exploratory scaling matrix.
- `long-warm/`: GPU visibility `3`; context 1024/2048, batch 1, one fresh process;
  one warm-up excluded from one measured iteration.
- `long-no-warm/`: GPU visibility `2`; context 1024/2048/4096 feasibility and KV
  capacity exploration. Its zero-warm-up throughput is invalid for framework ranking.
- `invalid-pilots/`: the first subagent-generated protocol. It is retained because it
  exposed two benchmark defects; none of its performance rows supports a claim.
- `fixed-cache-smoke.json`: hand-checkable Qwen Storage/active/capacity evidence after
  fixing KV accounting.
- `final-schema-smoke/`: final CLI/runner contract including top-logit, cache-prepare,
  steady-decode and end-to-end timing fields.

All matrices use the same pinned official model files and token cycles. microLLM uses
single-representation BF16 FFN/Attention weights with remaining FP32 paths; PyTorch uses
a full BF16 model. Ratios compare the currently executable policies, not identical dtype
residency.
