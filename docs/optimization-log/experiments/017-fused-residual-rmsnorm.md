# Experiment 017 — fused cached residual add and RMSNorm

Status: `keep`, with a recorded DeepSeek regression

## Hypothesis

In cached decoding, the first residual add in every block is immediately normalized for
the FFN, while the unnormalized sum is also needed by the second residual. One Kernel can
produce both outputs: `{left + right, rms_norm(left + right)}`. This should remove one
launch per layer without changing the training graph.

## Scope

- FP32 Tensor pair only;
- cached inference path only; autograd training remains the composed reference;
- returns both residual and normalized Tensor, so no residual value is discarded;
- CPU implementation composes `add` and `rms_norm`;
- HIP implementation uses one block-parallel reduction Kernel;
- independent PyTorch oracle checks both returned values.

## Correctness

- CPU debug: `157/157` pass;
- ASan/UBSan: `155/155` pass;
- MI300X/gfx942 HIP: `56/56` pass;
- Python/PyTorch operator oracle: `4/4` pass;
- zero H2D/D2H during focused HIP execution;
- cached MHA/GQA logits and official generated token IDs remain exact;
- training code and retained training measurements are unchanged.

## Three-process result

| Workload | Baseline median | Candidate median | Change | PyTorch ratio |
|---|---:|---:|---:|---:|
| Qwen generate | 142.01 | 154.60 token/s | +8.9% | 2.2029× |
| DeepSeek generate | 55.50 | 53.20 token/s | -4.2% | 0.8525× |
| Qwen train | unchanged | 112.43 token/s | 0% | 2.1905× |
| DeepSeek train | unchanged | 67.41 token/s | 0% | 2.5702× |

```text
score        1.784147 → 1.803226
```

DeepSeek is the current below-parity workload, so its regression is important. It is
below the protocol's 5% single-workload rejection threshold, but is not hidden or called
an improvement.

## Profiler explanation

Matched DeepSeek trace:

```text
all Kernel calls                    10,684 → 10,152
plain add calls                      1,120 →    588
fused add+RMSNorm calls                  0 →    532
total Kernel duration               116.17 → 114.99 ms
hipLaunchKernel API duration         65.26 →  59.70 ms
instrumented decode token/s          28.28 →  29.74
```

The profiler supports the launch-reduction hypothesis even though uninstrumented
DeepSeek process medians moved the other way. That disagreement is retained as a stable
warning about small fusion gains and process noise.

## Decision

`keep`: fixed score improves 1.1%, Qwen improves materially, no workload crosses the 5%
regression gate, the trace removes exactly the intended launches, and the public pair
operator has CPU/HIP/PyTorch evidence. A future shape policy must not be invented from
these two models alone; more widths are required before specializing the Kernel.

Raw evidence is in [017-data](017-data/README.md).
