# Experiment 052 evidence

- `pilot.jsonl`: one fresh Qwen/microLLM and PyTorch pair at context 512.
- `profile/`: candidate Qwen process-wide Kernel/HIP API aggregates.
- `comparison.json`: Experiment 051 baseline, candidate and discard calculation.
- `profile-summary.json`: exact aggregate and split-Kernel counts.

The candidate source was removed after the first process and profile both failed their
gates. The T=256 CPU/HIP Q/K/V correctness test passed before removal.
