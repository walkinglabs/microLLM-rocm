# In-place causal softmax for long Attention

Long hipBLASLt Attention now overwrites dead QK scores with causal probabilities. The public
out-of-place softmax remains unchanged; aliasing is restricted to the internal point where scores
have no later consumer and the cooperative kernel has completed all reads before writes.

Official Qwen/DeepSeek T2048 B8 logits are bit-identical. Peak memory falls exactly by one
`B*H*T*T*sizeof(float)` Tensor: 1.879 GB for Qwen and 1.611 GB for DeepSeek, while alternating
throughput pairs remain neutral-to-positive. The 16-shape survey has a 0.990x minimum throughput
ratio and all top tokens match.

See [Experiment 081](../optimization-log/experiments/081-inplace-causal-softmax.md).
