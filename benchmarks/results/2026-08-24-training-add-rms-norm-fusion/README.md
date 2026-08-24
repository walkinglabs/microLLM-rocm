# Training residual-add plus RMSNorm evidence

Experiment 210 tests one narrow training fusion. The temporary same-binary route changes
only the Attention residual add followed by the FFN RMSNorm. It keeps the residual sum as
an explicit Autograd node so the direct residual gradient and normalization gradient meet
before they are sent to both parents.

## Evidence

- `training.jsonl`: materialized/fused × Qwen/DeepSeek × three fresh processes;
- `summary.json`: alternating-order medians, loss, one observed parameter and peak memory;
- `*-diagnostics.json`: separate one-step graph diagnostics;
- `materialized-kernel-stats.csv` and `fused-kernel-stats.csv`: same-workload Qwen
  structural profiles;
- `profile-summary.json`: compact dispatch and Kernel-time attribution;
- `coverage-summary.json`: fresh gcovr 8.3 CPU source report;
- `verification.json`: final regression gates.

## Result

| Model | Materialized | Fused | Speedup | Peak ratio | Parameter gate |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 15,234.08 | 14,906.34 tok/s | 0.9785× | 1.000 | pass |
| DeepSeek Distill 1.5B | 6,324.03 | 6,311.17 tok/s | 0.9980× | 1.000 | fail |

The profile proves that the candidate really executed: 72 add launches and 72 standalone
RMSNorm launches become 72 fused launches. Total dispatches fall from 6,903 to 6,831, but
total Kernel time changes by only 0.045%. The eliminated work is too small relative to GEMM,
optimizer and backward reductions.

The model/CLI route is therefore removed. The standalone Autograd primitive remains because
CPU, HIP and PyTorch checks cover its two forward outputs and every branched gradient.
