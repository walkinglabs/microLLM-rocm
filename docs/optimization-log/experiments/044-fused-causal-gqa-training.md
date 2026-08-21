# Experiment 044 — fused full-sequence causal GQA training

Status: `keep`

## Problem

After Experiment 043, context 128 was dominated by the old full-sequence Attention graph:

```text
repeat K/V heads
→ Q @ Kᵀ
→ scale
→ materialize T×T scores
→ causal softmax
→ materialize T×T probabilities
→ probabilities @ V
```

Backward saved those large intermediates and ran the reverse graph. In the retained
context-128 trace, ordinary batched matmul took 158.9 ms and causal softmax
forward/backward took another 68.1 ms.

## Candidate

One causal GQA operator accepts `Q[B,H,T,D]` and `K/V[B,KV,T,D]`. It maps each query head
to its KV head directly, so repeated K/V tensors do not exist. A block computes one causal
query row, performs stable softmax in shared memory and writes its context vector without
materializing score or probability tensors.

Backward recomputes row probabilities, returns FP32 Q/K/V gradients and atomically reduces
shared KV-head gradients. The fallback retains the readable composed path when sequence is
over 4096 or head width is over 256.

## Correctness gates

- CPU fused forward/backward matches the composed graph.
- PyTorch operator oracle matches output and Q/K/V gradients.
- HIP covers MHA and GQA at T=1/3/32/128 with zero measured host payload transfers.
- Full tiny Transformer logits, every parameter gradient and trace topology align.
- CPU/HIP `247/247`, sanitizer `172/172`, PyTorch-enabled `177/177` pass.

The trace-alignment test initially failed because the PyTorch trace still exposed the old
six Attention sub-operations. Numerical checkpoints passed. The reference was corrected to
emit the same fused `causal_gqa_attention` boundary, then the trace gate passed.

## Official result

Baseline is retained Experiment 043. Each result is the median of three fresh microLLM and
three fresh PyTorch processes.

| Shape B×T | Before micro | Fused micro | Self speedup | Fused/PyTorch | Peak saved |
|---|---:|---:|---:|---:|---:|
| 1×3 | 31.17 | 33.45 tok/s | 1.073× | 0.771× | 1.3 MB |
| 2×3 | 61.37 | 64.57 tok/s | 1.052× | 0.682× | 2.7 MB |
| 1×32 | 268.63 | 304.38 tok/s | 1.133× | 0.614× | 21.6 MB |
| 1×128 | 659.23 | 802.70 tok/s | 1.218× | 0.438× | 185.6 MB |

![Fused causal GQA training](../assets/fused-causal-gqa-training.svg)

At context 128, Kernel dispatches fall `7678→6598` and aggregate Kernel time falls
`613.9→505.7 ms`. The fused forward/backward pair takes about 139.1 ms; it replaces the
old batched matmul, softmax, repeat and score/probability-storage path.

## Decision and next failure

Keep. Every official shape improves by at least 5%, every peak decreases, and the exact
graph/gradient contracts pass.

PyTorch parity is still open. The best selected row is 0.771× and context 128 is 0.438×.
After fusion, AdamW is again the largest single Kernel category (128.9 ms in the profiled
three-step process), followed by fused Attention backward (101.5 ms), Norm/bias reductions,
casts and remaining copies. The next candidate must start from this retained trace rather
than optimizing the removed T×T graph.
