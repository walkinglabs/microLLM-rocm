# Experiment 058 evidence

- `pilot.jsonl`: Qwen early keep/stop process.
- `formal/`: Qwen/DeepSeek × microLLM/PyTorch × three fresh processes at `1×512`.
- `fallback128.jsonl`: one Qwen process on the unchanged short-row implementation.
- `profile/`: retained Qwen Kernel and HIP API aggregates.
- `comparison.json`: model throughput, peak and fallback gates.
- `profile-summary.json`: forward/backward row-Kernel and whole-process deltas.

All official rows use BF16 Linear compute, FP32 master weights, one warm-up and two
measured steps. Raw JSONL/CSV is authoritative over rounded labels.
