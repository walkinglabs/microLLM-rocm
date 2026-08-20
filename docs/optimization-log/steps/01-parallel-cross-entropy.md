# Step 01 — parallel CrossEntropy forward/backward

Status: `complete — experiment 001 kept`

## Hypothesis

CrossEntropy's single-thread vocabulary loops are the dominant training bottleneck.
Block-parallel max/sum and gradient generation will materially improve Qwen/DeepSeek
train throughput without affecting generation.

## One-variable boundary

Only change CrossEntropy forward/backward execution. Do not simultaneously change dtype,
matmul, allocator or loss definition.

## Design

- one row per block;
- coalesced vocabulary loads;
- wave/block maximum reduction;
- wave/block exponential-sum reduction;
- FP32 accumulation;
- direct mean loss reduction across valid rows;
- direct `(p-y)/valid_rows` backward;
- preserve `ignore_index=-100`.

## Required tests

- classes: 2, 32, 8192, 151936;
- rows: 1, 3, 32;
- extreme positive/negative logits;
- all ignored and partially ignored targets;
- finite difference on small shapes;
- full PyTorch loss and gradient vector;
- CPU/HIP and sanitizer regression.

## Falsification

If CE Kernel time falls but end-to-end train throughput changes little, the hypothesis
that CE is the primary measured bottleneck is weakened; inspect synchronization and
transpose copies before adding more CE complexity.

## Keep gate

- existing numerical tolerance passes;
- CE is no longer the top 75% hotspot;
- Qwen and DeepSeek train ratios improve;
- generate ratios do not regress beyond noise.

## Actual result

All keep gates passed. Official train throughput improved 3.29× for Qwen and 2.29× for
DeepSeek; generation did not regress; the four-workload score improved 66.1%.
