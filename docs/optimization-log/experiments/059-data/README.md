# Experiment 059 evidence

- `pilot.jsonl`: Qwen early keep/stop process.
- `formal/`: two official models × two frameworks × three fresh processes at `1×512`.
- `fallback128.jsonl`: one Qwen short-row fallback process.
- `profile/`: retained Qwen Kernel/HIP API aggregates.
- `comparison.json` and `profile-summary.json`: machine-checked decision contracts.

Official rows use MI300X, BF16 Linear compute, FP32 master weights, one warm-up and two
measured steps. Raw JSONL/CSV is authoritative.
