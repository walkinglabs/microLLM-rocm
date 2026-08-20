# Experiment 038 — why BF16 training is slower than microLLM FP32

Status: `profile handoff`

## Paired structural trace

| Metric | FP32 | BF16 + FP32 master | Change |
|---|---:|---:|---:|
| Kernel dispatches | 2,886 | 3,247 | +361 |
| Kernel time | 25.15 ms | 26.02 ms | +3.4% |
| GEMM calls | 338 | 338 | unchanged |
| GEMM time | 6.39 ms | 5.06 ms | -20.9% |
| cast calls/time | 0 / 0 | 360 / 1.91 ms | added |
| engine allocation calls | 1,840 | 2,201 | +361 |

BF16 accelerates GEMM by 1.33 ms, but casts cost 1.91 ms and add host allocation/launch
work. The rest of forward/backward/AdamW remains FP32. This directly explains why
Experiment 037 is correct yet `8%–9%` slower than microLLM FP32.

The 336 FP32→BF16 casts are two per each of 168 Linear calls: activation and weight. The
remaining 24 BF16→FP32 casts are one small-M output fallback per layer.

## Next falsifiable candidate

Q/K/V share an activation. A graph op can cast that activation once while keeping three
independent FP32-master STE backward edges. It should remove exactly:

```text
Qwen:     24 layers × 2 saved casts × 5 measured steps = 240
DeepSeek: 28 layers × 2 saved casts × 5 measured steps = 280
```

The candidate is rejected if those calls disappear but paired three-process throughput
does not improve.
