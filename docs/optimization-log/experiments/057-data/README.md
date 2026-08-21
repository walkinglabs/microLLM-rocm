# Experiment 057 evidence

- `pilot.jsonl`: one Qwen process used for the early keep/stop decision.
- `formal/`: Qwen/DeepSeek × microLLM/PyTorch × three fresh processes at `1×512`.
- `fallback128.jsonl`: one Qwen short-sequence process on the unchanged fallback.
- `profile/`: retained Qwen Kernel and HIP API aggregates.
- `comparison.json`: official-model throughput, peak and fallback gates.
- `profile-summary.json`: exact removal/replacement accounting for saved-row backward.

The formal matrix alternates framework order and uses one warm-up plus two measured
training steps. Rounded Markdown and SVG labels are explanations; raw JSONL/CSV remains
the authoritative evidence.
