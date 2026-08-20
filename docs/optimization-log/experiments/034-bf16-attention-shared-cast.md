# Experiment 034 — single-representation BF16 Attention with shared Q/K/V cast

Status: `keep` — both decode rows improve; DeepSeek prefill stays within the 5% gate

## Hypothesis

Experiment 033 found GEMM at 67.64% of DeepSeek decode Kernel time. Four Attention Linear
weights per layer were still FP32. Converting those weights to BF16 should reduce persistent
memory and move more GEMM work onto the lower-precision path.

The first composition did this independently:

```text
FP32 normalized input → cast → Q GEMM
                      → cast → K GEMM
                      → cast → V GEMM
```

One official-model pilot preserved tokens and memory but regressed DeepSeek decode/prefill
by `2.1%/3.8%`. Qwen decode also regressed slightly. The result is retained in
`naive-pilot.jsonl`; a dtype change alone was not sufficient.

## Shared-cast design

`bf16_qkv_projection` performs:

```text
                       ┌→ BF16 Q weight → FP32 Q
FP32 input → BF16 once ├→ BF16 K weight → FP32 K
                       └→ BF16 V weight → FP32 V
```

Bias, RoPE, FP32 K/V cache and fused Attention remain unchanged. O projection uses a BF16
weight with FP32 input/output. This keeps the experiment to one numerical boundary.

`prepare_bf16_attention_inference()` transactionally converts Q/K/V/O weights only. Qwen
converts 96 tensors; DeepSeek converts 112. Attention biases stay FP32, and autograd/load
remain illegal after one-way inference preparation.

## Correctness gates

```text
CPU CTest                    163/163 pass
ASan/UBSan                   161/161 pass
HIP CTest                     63/63 pass
Python/PyTorch oracle           4/4 pass
official candidate rows          6/6 exact expected tokens
```

The operator oracle checks Q/K/V outputs and invalid shapes. The full model oracle rebuilds
BF16 Q/K/V/O and FFN weights in Python `torch`; CPU/HIP graph-free inference has a dedicated
zero-transfer comparison.

## Three-process result

Baseline is the retained Experiment 032 BF16-FFN policy.

| Model | Decode change | Prefill change | vs PyTorch BF16 decode/prefill | Resident weights |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | +2.90% | +6.89% | 1.213× / 1.300× | 1,260,477,952 B |
| DeepSeek Distill 1.5B | +1.98% | -2.72% | 0.533× / 1.017× | 4,487,960,576 B |

![BF16 Attention shared cast](../assets/bf16-attention.svg)

Full-vocabulary maximum differences versus a fresh FP32 reference are `0.16620` for Qwen
and `0.09691` for DeepSeek. Exact greedy IDs stay unchanged in all candidate processes.

Persistent weights fall another 88,080,384 bytes for Qwen and 308,281,344 bytes for
DeepSeek versus BF16 FFN only. The preparation peak does not rise above Experiment 031:
FFN conversion is committed before the smaller Attention transaction begins.

## Decision

Keep the generic shared-cast Q/K/V operator, model ownership API and official policy.
The candidate improves both decode rows, gives Qwen a useful prefill gain, saves memory,
and DeepSeek prefill remains inside the 5% single-workload gate.

It does not solve the user-level `>= PyTorch` target: DeepSeek decode is still only
`0.533×`. The next profiler must measure the retained candidate. Likely remaining boundaries
are the tied FP32 embedding/output head, FP32 O input/output conversion and launch count;
none is accepted without a new trace.
