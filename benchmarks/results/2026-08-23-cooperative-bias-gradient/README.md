# Cooperative bias-gradient evidence

Experiment 158 replaces one-thread-per-column serial row reduction with a
`32 columns × 8 row lanes` HIP block when `rows >= 32`.

- `operator-raw.jsonl`: 78 fresh-process Scalar/cooperative reports;
- `operator-summary.json`: 13 shape medians and complete-output errors;
- `baseline.jsonl` / `candidate.jsonl`: same-revision two-model training A/B,
  three fresh processes per model and policy;
- `training-summary.json`: speed, loss, parameter and peak-memory gates;
- `baseline-kernel-stats.csv` / `candidate-kernel-stats.csv`: same-workload
  rocprofv3 aggregates;
- `profile-summary.json` and `verification.json`: machine-readable decision.

The baseline was produced by temporarily raising the Auto threshold beyond the
executed row range, rebuilding the same worktree, and then restoring `rows >= 32`.
The final source and binary use the candidate policy.

Reproduce the operator matrix:

```bash
python3 benchmarks/single_gpu/bias_gradient_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_bias_gradient \
  --output-directory /tmp/bias-gradient --gpu 0 \
  --runs 3 --warmup 3 --repetitions 20
```
