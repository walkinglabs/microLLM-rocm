# Experiment 054 evidence

- `pilot.jsonl`: first Qwen context-512 candidate pair.
- `formal/`: Qwen/DeepSeek × two frameworks × three fresh processes.
- `fallback128.jsonl`: one retained short-sequence fallback process.
- `profile/`: retained Qwen candidate Kernel/HIP API aggregates.
- `comparison.json`: official keep gate and workspace boundary.
- `profile-summary.json`: Experiment 051 before/after timeline contract.

The profile's identified backward time includes row recompute, two batched GEMMs and two
GQA reductions. Matrix zero-fill time is left in total Kernel time rather than guessed per
operator.
