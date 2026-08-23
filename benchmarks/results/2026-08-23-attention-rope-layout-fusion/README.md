# Attention RoPE layout-fusion evidence

Experiment 162 removes the Q/K projection transpose materializations around fused
split-half RoPE during training. It does not change the mathematical operation.

Evidence files:

- `training.jsonl`: materialized/fused × Qwen/DeepSeek × three fresh processes;
- `summary.json`: medians, loss/parameter guards, memory and strided-copy diagnostics;
- `*-diagnostics.json`: one-step Autograd and Runtime layout attribution;
- `materialized-kernel-stats.csv` and `fused-kernel-stats.csv`: complete rocprofv3
  Kernel statistics for the same Qwen T512 workload;
- `profile-summary.json`: the exact rows used in the profile conclusion;
- `coverage-summary.json`: the complete post-change CPU coverage report;
- `verification.json`: regression gates and final decision.

The runner alternates policy order across process pairs. It accepts `--context 512` and
supplies 513 raw tokens because next-token training shifts one token into the target.
Each reported run therefore contains exactly 512 trained positions per step.

The fused graph reduces diagnostic strided copies by 60% for both models. Qwen throughput
is neutral (`0.9996×`) and DeepSeek improves (`1.0104×`). Peak engine memory falls by
48,234,496 and 102,760,448 bytes respectively. Complete PyTorch/CPU/HIP graph alignment
is required before these performance results are accepted.
