# 2026-08-21 — Qwen KV-cache prompt failure

The multi-pattern complete-logit runner falsifies a broad Qwen uniform-BF16 safety claim. Repeat,
rotated and ramp inputs pass, while constant inputs fail at context 32/512/2048. A two-FP32-layer
policy passes constant T512 but reaches RMSE 3.141 and changes a token at T2048.

First 4/8/12 FP32 layers do not rescue the long failure. Only an all-FP32 Cache passes that retained
counterexample. Uniform BF16 remains an explicit speed/memory path; the framework's FP32 default is
the required fallback when this strict gate is needed.

See [Experiment 071](../optimization-log/experiments/071-qwen-kv-prompt-failure.md) and
[raw evidence](../optimization-log/experiments/071-data/).

Final gates remain full CPU/HIP 268/268, ASan/UBSan 184/184 and PyTorch-enabled CPU 189/189.
