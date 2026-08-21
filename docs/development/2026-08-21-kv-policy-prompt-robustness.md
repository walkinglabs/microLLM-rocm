# 2026-08-21 — KV-policy prompt robustness

The complete-logit runner now supports deterministic repeat, rotated, constant and vocabulary-ramp
token patterns. The former one-FP32-layer DeepSeek policy passes only 9/14 challenge rows; constant
T512 reaches RMSE 2.995 and changes a greedy token.

The selected robust-strict policy keeps layers 0–3 FP32 and the remaining 24 layers BF16. It passes
14/14 prompt/context/batch rows, retains a 1.75x Cache reduction, and stays within roughly 3% of
uniform BF16 in the six-shape same-binary performance matrix. It remains explicit and pinned to the
tested checkpoint/pattern families.

See [Experiment 070](../optimization-log/experiments/070-kv-policy-prompt-robustness.md) and
[raw evidence](../optimization-log/experiments/070-data/).

Final gates: full CPU/HIP 268/268, ASan/UBSan 184/184 and PyTorch-enabled CPU 189/189.
