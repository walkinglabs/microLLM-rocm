# 2026-08-21 — batched long-sequence Attention forward

## Problem

After retaining saved probabilities and batched backward, the Qwen context-512 trace
still spent 272.52 ms in the readable fused forward Kernel. The implementation was
correct but each thread serially scanned the causal row.

## Change

`causal_gqa_attention_saved` now routes HIP sequences of at least 256 tokens through:

1. GQA K/V head expansion when needed;
2. query scaling;
3. strided-batched hipBLASLt `QK transpose-right`;
4. causal softmax;
5. strided-batched hipBLASLt `PV`.

Short sequences, unsupported dimensions and builds without hipBLASLt retain their old
paths. The public operator and autograd contracts did not change.

## Evidence

- MHA/GQA T=256 focused forward and Q/K/V backward comparison passed;
- Qwen/DeepSeek T=512 three-process medians improved 1.091x/1.165x;
- measured engine peak was byte-identical before and after for both models;
- Qwen T=128 fallback was 1.012x with byte-identical peak;
- forward-stage Kernel time fell 272.52 to 178.29 ms;
- total retained-process Kernel time fell 1283.85 to 1185.53 ms;
- dispatch count increased, ruling out launch-count reduction as the explanation.

Raw evidence, the falsification contract and the repository-owned SVG are in
[`experiments/056-data`](../optimization-log/experiments/056-data/) and
[`056-batched-attention-forward.md`](../optimization-log/experiments/056-batched-attention-forward.md).

## Remaining boundary

This is evidence for MI300X, BF16 training, batch 1 and context 512. It is not a claim
about every GPU or shape. The new trace points to saved-row backward and causal softmax
as the next long-sequence Attention costs.
