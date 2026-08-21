# Experiment 063 evidence

- `host-batch/`: Qwen/DeepSeek × B1/2/4/8 × microLLM/PyTorch on physical visibility 1,
  explicit host argmax control, one process.
- `batch/`: the same matrix and GPU with device row-wise argmax.
- `transfer-control.jsonl`: same-window Qwen B8 direct host/device rows with engine
  transfer calls and bytes.
- `profile-host/` and `profile-device/`: same Qwen B8 rocprof aggregates.
- `comparison.json` and `profile-summary.json`: machine-checked keep contracts.

Both modes use the same full-sequence model path, tokens, precision, warm-up and measured
iterations. Only `batch-argmax-mode` changes.
