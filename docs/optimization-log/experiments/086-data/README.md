# Experiment 086 data map

- `profile-summary.json`: curated phase-aware interpretation of the Release T2048 B1/B8 traces.
- `b1-*stats.csv`, `b8-*stats.csv`: direct rocprofv3 kernel, HIP API, allocation and copy
  aggregate tables.
- `d2h-pair-raw.jsonl`: three alternating baseline/candidate Release process pairs for B1/B8.
- `d2h-pair-summary.json`: medians, transfer counts and keep/discard decision inputs.

The full local traces are under `/tmp/microllm-exp086-profile-baseline/` and total about 319 MiB.
They are intentionally not duplicated into Git; the experiment document contains the exact command
and the committed aggregate tables are sufficient to rerun and compare the trace.

The D2H candidate was fully reverted after B8 failed the performance gate. Its data is retained as
a counterexample, not as the current implementation.
