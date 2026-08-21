# 2026-08-21 — explicit mixed-layer KV-cache policy

`KVCache` now supports one FP32/BF16 dtype per model layer. The model reads the policy at each layer;
the public generator accepts the same vector, and the CLI exposes explicit FP32 layer indices over a
base Cache dtype. Uniform constructors and FP32 defaults remain source-compatible.

The selected pinned DeepSeek policy keeps layer 1 FP32 and the other 27 layers BF16. It changes the
only Experiment 065 Release failure from RMSE 0.058645 to 0.039543. The complete 12-shape matrix
passes logits, finite, top-token and suffix gates while retaining a 1.920x/1.931x Cache reduction for
Qwen/DeepSeek.

This is not an automatic default because the selected layer is checkpoint-specific and Cache is
3.57%–4.17% larger than uniform BF16. Experiment 069 later invalidates this experiment's
cross-window long-batch slowdown attribution with a same-binary paired matrix. See
[Experiment 067](../optimization-log/experiments/067-mixed-layer-kv-policy.md) and
[raw evidence](../optimization-log/experiments/067-data/).

Experiment 070 later demotes layer 1 to a fixed-prompt result and selects layers 0–3 for the broader
prompt challenge. Final gates at this node were full CPU/HIP 267/267, ASan/UBSan 183/183 and
PyTorch-enabled CPU 188/188.
