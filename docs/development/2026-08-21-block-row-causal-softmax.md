# 2026-08-21 — cooperative long-row causal softmax

## Delivered

Causal softmax forward and backward now use one cooperative 256-thread HIP block per
row for sequences of at least 256 tokens. Short rows retain the old one-thread-per-row
implementation.

## Evidence

- standalone softmax and complete T=256 MHA/GQA forward/all-gradient tests pass;
- Qwen/DeepSeek T512 three-process medians improve 1.302x/1.196x;
- measured peak is byte-identical for both models;
- T128 fallback is 1.002x with byte-identical peak;
- forward/backward softmax Kernel time improves 4.253x/4.801x;
- full retained-process Kernel time improves 1.279x with identical dispatch count.

See the [experiment report](../optimization-log/experiments/058-block-row-causal-softmax.md)
and [raw evidence](../optimization-log/experiments/058-data/).

## Boundary

The threshold is measured for MI300X context 128/512. Other sequence lengths and Radeon
need their own matrix. RMSNorm weight-gradient reduction is now the largest Kernel.
