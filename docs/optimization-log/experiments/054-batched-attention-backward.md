# Experiment 054 — batched-GEMM long-sequence Attention backward

## Design

For T≥256 with hipBLASLt available:

1. one row Kernel recomputes causal probabilities and scaled score gradients, writes the
   lower triangle, and computes Q gradient;
2. two strided-batched GEMMs compute `dSᵀ @ Q` and `Pᵀ @ dO` for every query head;
3. the existing repeat-head backward reduces query-head K/V gradients to GQA KV heads.

Both T×T matrices are zero-filled before row writes; this is the causal contract caught by
the first T=256 test. T<256 and builds without hipBLASLt retain the atomic Kernel.

## Correctness

- MHA/GQA Q/K/V gradients match CPU at T=256;
- upper-triangle future contributions are zero;
- FP32 batched GEMM error is at most about 1.1e-6;
- optimizer payload transfer remains zero;
- T=128 fallback has the same peak and `1.008×` single-process throughput.

## Official result

| Model | T=512 before | After | Self speedup | PyTorch ratio | Peak change |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 812.45 | 1103.05 tok/s | 1.358× | 0.130× | 1.000× |
| DeepSeek Distill 1.5B | 400.15 | 546.07 tok/s | 1.365× | 0.116× | 1.000× |

![Batched Attention backward](../assets/batched-attention-backward.svg)

The theoretical workspaces are about 28 MiB for Qwen and 24 MiB for DeepSeek. They do not
raise the measured training peak because a larger live allocation already determines it.

## Retained profile

Qwen process Kernel time falls `1.946→1.442 s` (`1.350×`). The 985.61 ms atomic backward
is replaced by 473.91 ms row recompute, 1.52 ms batched GEMM and 4.49 ms GQA reduction;
matrix fills remain conservatively inside total time. Dispatches rise 4.3% and HIP API
calls rise 5.1%, yet device and end-to-end time improve.

## Remaining gap

The selected PyTorch ratios are still only 0.130×/0.116×. Forward Attention remains a
serial row Kernel, and the row recompute stage now dominates backward. A flash-style tile
that reuses Q/K/V and softmax statistics is still required for parity.
