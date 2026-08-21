# Register-cached causal softmax

For sequence lengths up to 2048, the retained HIP kernel keeps each thread's eight exponentials in
registers until denominator reduction completes. It preserves the exact reduction order and both
official Qwen/DeepSeek T2048 B8 logits are bit-identical to the independent reference.

An initial cross-window Qwen result looked 11.1% slower, but the reference binary drifted in the
same period. Alternating same-window binaries produced 1.046x Qwen and 1.022x DeepSeek median pair
ratios. Qwen softmax device time falls 14.9%; code-object metadata reports no private segment or
register spill. A DeepSeek T512 B1 single-pair outlier also failed to reproduce across three pairs.

See [Experiment 079](../optimization-log/experiments/079-register-softmax.md).
