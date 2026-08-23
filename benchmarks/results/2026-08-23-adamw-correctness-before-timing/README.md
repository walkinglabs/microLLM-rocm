# AdamW correctness-before-timing evidence

This directory records Experiment 157 on one visible gfx942 MI300X device.

- `raw.jsonl`: 15 fresh-process operator reports, three per case;
- `summary.json`: process-median Event timing and complete-state gate summary;
- `accepted-report.json`: explicit acceptance of the safe unaligned Scalar result;
- `accepted-cache.jsonl`: exact environment/alignment/mode cache entry produced by the CLI;
- `end-to-end.jsonl`: three tiny FP32 T128/B8 training regressions;
- `verification.json`: machine-readable keep/discard decision.

Every supported candidate compares every element of the updated FP32 parameter, first
moment, second moment, and optional BF16 mirror before any timing is recorded. The
unaligned Vectorized counterexample is unsupported and has zero Event and wall timing.
No aligned case reaches the 1.05 operator keep gate, so the infrastructure is retained
while `Auto` continues to select Scalar unless an exact explicitly accepted cache entry
says otherwise.

Reproduce the matrix:

```bash
python3 benchmarks/single_gpu/adamw_autotune_matrix.py \
  --binary build/hip-release/benchmarks/microllm_tune_adamw \
  --output-directory /tmp/adamw-autotune \
  --gpu 0 --runs 3 --warmup 3 --repetitions 20
```
