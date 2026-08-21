# Experiment 043 evidence

- `candidate/raw.jsonl`: four shapes × two frameworks × three fresh processes.
- `candidate/summary.json`: per-framework medians.
- `comparison.json`: candidate versus Experiment 042 retained baseline.
- `microbench.jsonl`: exact Qwen weight-gradient shapes, readable and hipBLASLt.
- `profile/`: aggregated context 32 before/after and context 128 before traces.

The raw rocprof traces are much larger and are not committed. Reproduce aggregation with
`scripts/summarize_rocprof.py`. Reproduce the official candidate with the Experiment 042
command after checking out the retained source; use a different output directory so the
baseline is not overwritten.
