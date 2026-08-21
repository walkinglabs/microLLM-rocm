# Experiment 090 data map

- `summary.json`: keep decision, alternating medians and transfer contracts.
- `t2048-pair-*`: DeepSeek B1/B8 three-pair Exp087/Exp090 comparison.
- `qwen-t512-b8-pair-*`: targeted three-pair recheck of the noisy single-process row.
- `qwen-matrix-*`, `deepseek-matrix-*`: candidate/PyTorch six-shape surveys.
- `gates.json`: operator, generation, CPU/HIP and sanitizer gates.

The fast path applies only to HIP greedy generation with no stop-token policy. Sampling and
early-stop paths preserve their per-step host decisions. Total copied token bytes do not change;
the number of host synchronization boundaries falls from one per token to one per generation run.
