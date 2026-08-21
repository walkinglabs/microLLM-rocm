# Block-0 Attention/FFN drift split

The inference layer trace now adds detailed values only inside block zero: attention norm, Q/K/V,
RoPE, value, context/output, residual, FFN norm and FFN output. Inactive tracing and the other 27
blocks retain their prior paths.

Three official DeepSeek B1/B2 pairs show exact values through FFN norm. The first nonzero stage is
`inference.blocks.0.ffn.output`, max 0.0013504 and relative L2 0.00007269. Attention projections are
exact despite their changed flattened M shape, so the remaining investigation is specific to the
fused BF16 FFN.

See the [beginner guide](../dev/block0-drift.zh-CN.md) and
[Experiment 107](../optimization-log/experiments/107-block0-drift.md).
