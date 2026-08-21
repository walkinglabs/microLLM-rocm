# Experiment 050 evidence

- `load-smoke.jsonl`: Qwen and DeepSeek with load-specific bytes, calls, current and peak.
- `deepseek-formal/raw.jsonl`: four shapes × two frameworks × three fresh processes.
- `deepseek-formal/summary.json`: generated medians after streaming integration.
- `comparison.json`: old/new load sources, PyTorch ratio, training non-regression and safety.

The Qwen old-load control comes from Experiment 047's matched raw because it used the same
pre-streaming loader and recorded `load_ms`. The DeepSeek control is Experiment 045.
