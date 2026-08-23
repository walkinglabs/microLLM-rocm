# Complete Attention context-layout fusion evidence

Experiment 164 connects the interleaved P×V, dP and dV GEMMs to causal GQA Autograd.
Q/K remain BHTD; Value and context remain BTHD from projection through output Linear.

- `training.jsonl`: context materialized/fused × Qwen/DeepSeek × three fresh processes;
- `summary.json`: medians, loss/parameter, memory and exact layout counters;
- `*-diagnostics.json`: one-step source/layout attribution;
- `materialized-kernel-stats.csv` and `fused-kernel-stats.csv`: complete same-workload
  Qwen rocprofv3 Kernel statistics;
- `profile-summary.json`: selected profile facts and ratios;
- `coverage-summary.json`: complete post-change CPU coverage;
- `verification.json`: all correctness, performance and regression gates.

The same-binary runner holds RoPE layout fusion enabled and changes only
`--attention-context-layout-fusion`. Qwen/DeepSeek T512 improve `1.0336×/1.0256×`,
peak memory falls by 100,401,152/205,520,896 bytes, and the previously attributed
strided-copy counter reaches exactly zero on both models.
