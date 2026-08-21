# BF16 paired Value load rejection

Experiment 092 has one thread accumulate two neighboring BF16 Value columns while preserving each
column's position order. DeepSeek T2048 B1/B8 complete logits are bit exact, but alternating
Release medians are 0.988x/0.989x. The candidate is fully reverted.

Together with Experiments 088–091 and the earlier thread/query-staging failures, this closes local
scalar cached-Attention rewrites. A future attempt needs a score-level gate and a larger wave/MFMA
or online-softmax algorithmic change.

See [Experiment 092](../optimization-log/experiments/092-bf16-paired-value-load-discard.md).
