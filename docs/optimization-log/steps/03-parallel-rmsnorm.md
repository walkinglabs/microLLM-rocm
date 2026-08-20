# Step 03 — block-parallel RMSNorm

Status: `complete` — Experiment 003, `keep`

## Hypothesis

One-thread-per-row RMSNorm causes the 37.7% inference and 10.2% training hotspot.

## Design

- one block per row;
- vectorized items per thread;
- wave reduction then block reduction;
- FP32 square sum and weighted dot;
- forward writes normalized × weight in the same Kernel;
- backward emits input grad plus block partial weight grad;
- second stage reduces weight-grad partials.

## Required tests

- widths 16, 384, 512, 896, 1536;
- rows 1, 3, 32;
- epsilon variants from real configs;
- extreme values and zeros;
- FP32 PyTorch parity and finite difference;
- future BF16 accumulator test scaffold.

## Falsification

If Kernel time improves but end-to-end does not, Kernel launch, allocation or forced
synchronization is hiding the gain.

## Keep gate

- RMSNorm no longer dominates Qwen inference;
- Qwen/DeepSeek train and generate ratios improve;
- no atomic hot spot replaces the serial row loop.

## Measured result

```text
Qwen train                 38.77 → 71.06 token/s
Qwen generate              35.35 → 57.32 token/s
DeepSeek train             22.36 → 47.91 token/s
DeepSeek generate          10.15 → 18.60 token/s
four-workload score       0.479227 → 0.885816
RMSNorm Kernel time        75.85 ms → 1.55 ms
RMSNorm Kernel share        64.31% → 3.59%
```

Backward first computes one inverse RMS per row, then a second Kernel reduces each
weight-gradient column across rows. This avoids turning the old serial loop into an
atomic-contention problem.

See [Experiment 003](../experiments/003-parallel-rmsnorm.md) for correctness gates,
raw results and the exact scope of the PyTorch comparison.
