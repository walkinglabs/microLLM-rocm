# Experiment 016 — fused Q/K projection bias and split-half RoPE

Status: `keep`

## Observation

The latest DeepSeek FP32 decode trace attributed 4.92% of Kernel time to 1,680
`add_bias` launches and 2.89% to 1,120 split-half RoPE launches. Q and K always apply
these two operations consecutively; V uses bias but not RoPE.

## Hypothesis

One Kernel can read the raw Q/K projection plus its head-aware bias and directly write
the rotated result. This removes an intermediate Tensor, one allocator retirement Event
and one Kernel launch per Q or K without changing model math.

## Scope

- FP32 `[B,H,T,D]` input and `[H*D]` bias;
- split-half Qwen RoPE only;
- Q and K projections only; V bias is unchanged;
- CPU reference and standalone autograd operation are public and tested;
- backward uses inverse RoPE then reduces the FP32 pre-RoPE gradient into bias;
- interleaved RoPE and bias-free models keep the old path.

## Correctness gates

- CPU debug: `156/156` pass;
- ASan/UBSan: `154/154` pass;
- MI300X/gfx942 HIP: `55/55` pass;
- Python/PyTorch operator and graph oracle: `4/4` pass;
- fused/composed CPU forward, input gradient and bias gradient match;
- complete Transformer CPU/HIP graph alignment passes;
- cached MHA/GQA logits and official Qwen/DeepSeek top logits/tokens are exact;
- Qwen/DeepSeek multi-step loss and observed AdamW parameter remain unchanged.

## Paired three-process result

Each process uses 2 warm-ups and 5 measured iterations. The baseline was independently
built from commit `914b2d6` and run in the same time window as the candidate.

| Workload | Baseline median | Candidate median | Change | PyTorch ratio |
|---|---:|---:|---:|---:|
| Qwen train | 112.32 | 112.43 token/s | +0.1% | 2.1905× |
| Qwen generate | 124.88 | 142.01 token/s | +13.7% | 2.0235× |
| DeepSeek train | 67.15 | 67.41 token/s | +0.4% | 2.5702× |
| DeepSeek generate | 52.05 | 55.50 token/s | +6.6% | 0.8894× |

```text
paired baseline score       1.698264
candidate score             1.784147
historical running best     1.770568
```

The paired baseline is slower than the historical baseline, especially for Qwen
generation. The report therefore shows both comparisons: paired runs establish that the
fusion helps in the same time window, while the fixed PyTorch ratios decide the new
running-best score. No single process is selected by hand.

## Allocation and profiler result

Official comparison allocation calls:

```text
Qwen generation       11,165 →  9,965  (-1,200)
DeepSeek generation   48,585 → 43,265  (-5,320)
```

Matched DeepSeek profiler:

```text
all Kernel launches                 11,804 → 10,684
add_bias                            1,680 →    560
split-half RoPE                     1,120 →      0
fused bias+split-half RoPE              0 →  1,120
total Kernel duration              120.26 → 116.17 ms
hipLaunchKernel API duration        77.80 →  65.26 ms
hipEventRecord calls               10,057 →  8,993
```

Exactly 1,120 Kernel launches disappear. Attention, RMSNorm and GEMM call counts remain
unchanged, supporting the intended causal explanation.

## Decision

`keep`. The candidate improves the fixed score, has no paired workload regression,
reduces allocation/Event churn, and preserves forward, backward, optimizer and official
token evidence. DeepSeek generation remains below PyTorch, so the next experiment must
target the remaining GEMM/launch structure rather than claim completion.

Raw evidence is in [016-data](016-data/README.md).
