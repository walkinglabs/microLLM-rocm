# Forward-only GQA Value broadcast evidence

Experiment 171 is the final zero-stride variant. Only width>=128 forward P×V broadcasts
V; backward restores expanded V and one H-batched dP.

- `training.jsonl` / `summary.json`: disabled/enabled × Qwen/DeepSeek × three fresh T512
  processes;
- `baseline-kernel-stats.csv` / `forward-kernel-stats.csv`: complete DeepSeek profile;
- `profile-summary.json`: repeat, dispatch and Kernel-time attribution;
- `coverage-summary.json`: post-change CPU coverage;
- `verification.json`: final zero-stride closure and regression gates.

DeepSeek reaches only `1.0009×`, saves 56 allocations, does not lower peak, and changes the
fixed parameter because P×V accumulation order differs. Qwen D64 is not routed and measures
`0.9822×` process drift. Profile removes 84 repeat launches and adds 84 GEMM launches;
dispatches remain 8,058 and Kernel time rises 0.88%. All zero-stride model policies default
false; only tested primitives remain.
