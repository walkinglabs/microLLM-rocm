# Inference micro-fusion saturation audit

The accepted direct-BF16 Q/K route is followed by two consecutive rejected
candidates: causal-softmax block-size tuning and BF16 V cast/repeat fusion.
Current profiles and perfect-elimination bounds show that further one-launch
changes cannot plausibly close the remaining gap across both models.

The inference micro-fusion track is closed. Future Attention work must be a
separate MFMA/rocWMMA tiled online design with new correctness, memory and
end-to-end acceptance gates. See
[Experiment 209](../optimization-log/experiments/209-post-bf16-qk-saturation.md).
