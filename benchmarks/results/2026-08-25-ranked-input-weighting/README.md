# Ranked uneven-input weighting

Experiment 276 runs from clean revision `8e99c53` with tiny two-rank training.
Rank0 has one row (4 valid tokens); rank1 has two rows (8 valid tokens).

The default equal-only policy first exchanges token counts, detects the mismatch,
and fails before parameter collectives. Both workers return 1; the group exits in
3.056 seconds without hanging.

The explicit token-weighted policy computes an average of 6 tokens/rank and uses:

```text
rank0 gradient scale = 4 / 6 = 0.666666687
rank1 gradient scale = 8 / 6 = 1.333333373
RCCL average(scaled gradients) = global B3 mean gradient
```

After three steps, rank Max/RMS are zero. Relative to the concatenated CPU B3
reference, parameter Max/RMS are 8.18e-8/8.79e-9 and the maximum difference
between row-weighted local loss and global loss is 1.94e-7.

Weighted ready-overlap remains unsupported because buckets may be enqueued before
post-backward scaling. Synchronous per-parameter/bucket/view paths can use the
explicit weighting mode.
