# Step 02 — transpose-aware GEMM without materialization

Status: `planned`

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
- cache descriptor/layout by exact key;
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
