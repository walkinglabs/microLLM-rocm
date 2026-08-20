# Experiment 039 — shared-QKV cast in the training graph

Status: `discard` — allocation hypothesis succeeds, throughput hypothesis fails

## Candidate

One BF16 Q/K/V graph operation shared the forward activation cast, then created three
ordinary graph outputs. Each output retained its own FP32-master input/weight gradient
formula. Focused forward and all four leaf gradients exactly matched three independent
BF16 STE matmuls.

## Measured result

Baseline is Experiment 037 BF16 training.

| Model | Candidate | vs BF16 baseline | vs micro FP32 | vs PyTorch AMP | Peak |
|---|---:|---:|---:|---:|---:|
| Qwen | 134.90 tok/s | 0.973× | 0.894× | 3.037× | unchanged |
| DeepSeek | 74.73 tok/s | 1.009× | 0.914× | 2.606× | unchanged |

![Shared QKV training candidate](../assets/bf16-training-qkv-discard.svg)

Allocation calls fall exactly as predicted: Qwen `11,000→10,760`, DeepSeek
`12,825→12,545` over five measured steps. Both loss trajectories and all parameter updates
remain finite. Yet the two-model geometric throughput ratio is about `0.991×`, and Qwen
regresses 2.7%.

## Decision

Discard the autograd ValueTriple/API, model branch and tests. The inference-only
`ops::bf16_qkv_projection` remains because its own official inference matrix passed.

Fewer casts/allocations alone do not pay for additional graph nodes and scheduling in this
short training shape. The next training candidate must form a larger continuous FFN island
or address forward-weight lifecycle; it cannot reintroduce this graph API without new
evidence.
