# BF16x2 cached-Key load rejection

Experiment 088 tests one vector-read candidate inside BF16 fused cached Attention. Four focused HIP
tests pass, but DeepSeek T2048 complete cached logits fail: B1 max/RMSE is 0.05650/0.01323 and B8
is 11.978/1.528 with a third-token divergence. No performance result is accepted.

The implementation is fully reverted. Future packed loads must restore two public `hip_bfloat16`
values from raw 16-bit lanes rather than reinterpret them as an internal vector type.

See [Experiment 088](../optimization-log/experiments/088-bf16x2-key-load-discard.md).
