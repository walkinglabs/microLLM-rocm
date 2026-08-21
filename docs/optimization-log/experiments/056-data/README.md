# Experiment 056 evidence

- `pilot.jsonl`: first Qwen pair after routing long-sequence forward through batched GEMM.
- `formal/`: Qwen/DeepSeek × microLLM/PyTorch × three fresh processes at `1×512`.
- `fallback128.jsonl`: one Qwen short-sequence process proving the old path stays active.
- `profile/`: retained Qwen Kernel and HIP API aggregates.
- `comparison.json`: model throughput, memory and fallback decision.
- `profile-summary.json`: exact Experiment 055/056 forward and whole-process deltas.

All measurements use BF16 Linear compute with FP32 master weights, one warm-up step and
two measured steps. The formal runner alternates framework order. The raw files, not the
rounded chart labels, are authoritative.
