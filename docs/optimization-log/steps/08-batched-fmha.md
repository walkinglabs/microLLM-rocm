# Step 08 — batched GEMM and fused Attention

Status: `planned`

## Hypothesis

After KV/layout fixes, readable rank-4 QK/PV matmul and materialized score/probability
will dominate as context grows.

## Stages

1. strided-batched GEMM reference candidate;
2. direct KV-head mapping;
3. block-parallel causal softmax;
4. fused prefill Attention;
5. fused decode Attention;
6. optional Composable Kernel FMHA backend.

## Required matrix

```text
B      1, 2
T      1, 32, 128, 512
heads  MHA and GQA
dtype  FP32 first; BF16 separate
```

## Correctness

- full attention output against PyTorch;
- causal future values/gradients zero;
- forward/backward finite difference on tiny shapes;
- cached/full every prefix;
- fallback for unsupported head dimensions.

## Falsification

If fused Attention wins microbench but short HF end-to-end does not move, the workload
is dominated by projection, Norm or launch overhead; retain only with long-context
evidence and manageable complexity.

## Keep gate

Both operator and end-to-end context curves improve, peak score/probability memory drops,
and readable Attention remains available.
