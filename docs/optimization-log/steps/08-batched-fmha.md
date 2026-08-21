# Step 08 — batched GEMM and fused Attention

Status: `in progress` — cached decode stage kept in Experiment 009

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

## Cached decode stage result

The first retained stage fuses cached score, stable softmax and context for FP32
sequence `<=4096`; longer sequences keep the readable fallback.

```text
Qwen generation median      134.87 → 142.25 token/s (+5.5%)
DeepSeek generation median   49.05 → 53.04 token/s (+8.1%)
Qwen 32/128/512 curve        +18.5% / +18.5% / +57.9%
Qwen 1-token curve           -7.8% stable failure
```

Prefill, training backward and BF16 remain unfinished; this step is not marked complete.

Experiment 019 also tested 64/128-thread blocks for head widths 64/128. Both official
generation medians regressed, so the retained fused decode Kernel remains at 256 threads.

Experiment 044 completes the first full-sequence training forward/backward stage for FP32
Attention activations. Direct GQA mapping and row-recomputed softmax improve all four Qwen
BF16-training shapes by 1.052×–1.218× and reduce context-128 peak by 185.6 MB. T=512,
batch-long and DeepSeek rows remain required before this step can be marked complete.

Experiment 051 supplies the missing long-sequence boundary. Qwen/DeepSeek context 512 are
correct but reach only 0.0978×/0.0832× PyTorch throughput. Qwen profiling attributes
64.50% of Kernel time to causal GQA, with backward alone at 50.64%. The next candidate
must replace atomic K/V accumulation for long sequences, report the T×T workspace cost,
and preserve the retained T=128 path as a fallback.

Experiment 052 tests the first long-sequence alternative. It writes probability and
score-gradient matrices, then gives each K/V output element a non-atomic reduction. Q/K/V
correctness passes, but backward time rises 34% and Qwen T=512 throughput falls 15%.
The scalar rescan implementation is removed. The remaining credible space is tiled
matrix/flash-style backward with score-tile reuse.

Experiment 053 supplies strided-batched hipBLASLt, and Experiment 054 uses it for K/V
gradients after one causal row-recompute pass. Qwen/DeepSeek T=512 improve 35.8%/36.5%
with unchanged measured peak; T=128 remains on the old path. The candidate is retained.
The row recompute Kernel (and the analogous forward row Kernel) is now the remaining
Attention target.

Experiment 055 saves long-sequence forward probabilities in the autograd closure. Qwen and
DeepSeek improve another 13.2%/15.0%, with a fixed +336 MiB measured peak cost; T=128
retains recomputation. The policy is kept as an explicit speed/memory trade-off. Forward
and saved-row backward are now the two similarly sized hotspots.
