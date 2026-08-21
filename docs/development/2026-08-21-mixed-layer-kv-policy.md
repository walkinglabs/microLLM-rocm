# 2026-08-21 — explicit mixed-layer KV-cache policy

`KVCache` now supports one FP32/BF16 dtype per model layer. The model reads the policy at each layer;
the public generator accepts the same vector, and the CLI exposes explicit FP32 layer indices over a
base Cache dtype. Uniform constructors and FP32 defaults remain source-compatible.

The selected pinned DeepSeek policy keeps layer 1 FP32 and the other 27 layers BF16. It changes the
only Experiment 065 Release failure from RMSE 0.058645 to 0.039543. The complete 12-shape matrix
passes logits, finite, top-token and suffix gates while retaining a 1.920x/1.931x Cache reduction for
Qwen/DeepSeek.

This is not a default optimization. DeepSeek steady decode regresses at most 2.43%, but T2048 B8
cache preparation and end-to-end time regress 27.9%/13.4%. The policy remains an explicit strict
logit-accuracy trade-off. See
[Experiment 067](../optimization-log/experiments/067-mixed-layer-kv-policy.md) and
[raw evidence](../optimization-log/experiments/067-data/).

Final gates: full CPU/HIP 267/267, ASan/UBSan 183/183 and PyTorch-enabled CPU 188/188.
