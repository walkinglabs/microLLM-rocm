# Step 02 — transpose-aware GEMM without materialization

Status: `complete` — Experiment 002, `keep`

## Hypothesis

Eliminating full-weight `transpose().contiguous()` will remove the 43.4% Qwen inference
strided-copy hotspot and reduce training allocation/copy overhead.

## One-variable boundary

Change matrix layout semantics and hipBLASLt submission only. Keep mathematical model,
FP32 dtype, allocator and Attention decomposition unchanged.

## Design

- extend matmul contract with transA/transB or general strides;
- submit logical transpose through hipBLASLt layout/op flags;
- use it for tied output projection;
- use it for both sides of Linear backward;
- leave descriptor/layout caching for a later single-variable experiment;
- no permanent duplicate tied weight in the first version.

## Required tests

- NN, NT, TN, TT hand matrices;
- rectangular and batched-representative shapes;
- non-contiguous view boundaries;
- Qwen tied output complete logits;
- every Linear gradient versus PyTorch;
- allocation counter proves no vocabulary-sized transpose buffer.

## Falsification

If strided-copy disappears but throughput does not improve, host allocator/synchronization
or RMSNorm dominates more than Kernel percentages implied.

## Keep gate

- output projection copy absent from rocprof;
- Qwen inference peak drops or stays bounded;
- all four throughput rows non-regressing;
- no hidden full-weight cache that doubles persistent model memory.

## Measured result

```text
Qwen train                 24.03 → 38.77 token/s
Qwen generate              18.85 → 35.35 token/s
DeepSeek train             13.30 → 22.36 token/s
DeepSeek generate          10.05 → 10.15 token/s
four-workload score       0.318328 → 0.479227
strided-copy Kernel time   62.33 ms → 2.16 ms
strided-copy calls          1302 → 624
```

Qwen inference peak engine memory fell from about 2.35 GiB to 1.84 GiB. The unchanged
DeepSeek generation row is useful evidence: this change accelerates graphs that actually
contain the removed tied-weight copy; it is not a universal generation speedup.

Full raw evidence and the exact test result are recorded in
[Experiment 002](../experiments/002-transpose-aware-gemm.md).
