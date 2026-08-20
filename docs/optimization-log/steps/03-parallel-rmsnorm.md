# Step 03 — block-parallel RMSNorm

Status: `planned`

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
