# Experiment 033 — isolate the remaining DeepSeek BF16 decode bottleneck

Status: `profile handoff` — no optimization claim

## Why this profile exists

After Experiment 032, Qwen decode/prefill and DeepSeek prefill exceed the fixed PyTorch
full-BF16 reference. DeepSeek decode remains `0.522×`. The new `--workload decode` mode
removes full-sequence prefill from the traced application path.

## What the trace says

The trace contains 10,038 Kernel dispatches and 237.60 ms of instrumented Kernel time.
The prompt plus generation executes 19 cached token forwards, so `532 = 28 layers × 19`.

| Category | Calls | Kernel time | Share |
|---|---:|---:|---:|
| hipBLASLt GEMM | 3,743 | 160.71 ms | 67.64% |
| fused cached Attention | 532 | 24.28 ms | 10.22% |
| BF16 casts, including preparation | 1,148 | 14.63 ms | 6.16% |
| fused Q/K bias + RoPE | 1,064 | 7.55 ms | 3.18% |
| RMSNorm | 551 | 6.62 ms | 2.78% |

The 3,743 GEMMs decompose exactly as:

```text
28 layers × 19 cached forwards × 7 Linear GEMMs = 3,724
+ 19 tied output-head GEMMs
= 3,743
```

Only the three FFN GEMMs use BF16 weights. Q/K/V/O Attention projections and the tied
output head still use FP32, while the PyTorch competitive reference stores and computes
the whole model in BF16.

HIP API totals include checkpoint load and first-use setup, so `hipMemcpy` and
`hipModuleLoad` are not decode hotspot evidence. Even after excluding those, launch/API
churn is secondary to GEMM Kernel time. Experiment 021 already showed that blindly caching
device selection can regress every workload.

## Next contract

The next candidate may change only inference Linear precision ownership:

1. add single-representation BF16 weights for Attention projections;
2. keep Norm/softmax accumulation FP32;
3. avoid three separate casts of the shared normalized Attention input;
4. keep K/V cache FP32 in the first attempt so Attention math does not change twice;
5. compare exact tokens, full logits, persistent/preparation memory and three-process speed;
6. reject if DeepSeek decode does not improve or any already-green row regresses over 5%.

This is evidence for a broader BF16 Linear island, not permission to flip every Tensor to
BF16 at once.
