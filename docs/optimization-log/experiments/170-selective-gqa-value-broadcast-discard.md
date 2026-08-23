# Experiment 170 — selective full Value broadcast still loses

## Candidate

Experiment 169 showed opposite operator results: D64 Qwen loses while D128 DeepSeek wins.
The graph therefore routes only when all are true:

```text
HIP + hipBLASLt + T>=256 + repeats>1 + head width>=128
```

Forward P×V reads original V with zero batch stride. Backward dP does the same for
`dO×Vᵀ`. K expansion and dK remain unchanged. dV still produces expanded head gradients
and reduces them, preserving its known path.

## Correctness

- CPU/PyTorch dP compares against explicit repeated V;
- HIP B2 dP/output make zero payload transfers;
- a T256/D128 saved Attention test compares output, probabilities and Q/K/V gradients
  against the expanded GPU control;
- Qwen D64 is structurally excluded;
- same-binary false restores all Value expansions.

## Official T512 A/B

Rejected plan/alpha/paired policies stay false. Only
`--attention-gqa-value-broadcast` changes.

| Model | Disabled | Selective enabled | Speedup | Allocations saved | Peak ratio | Parameter equal |
|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-0.5B D64 | 14,916.04 | 14,837.80 tok/s | 0.9948× | 0 | 1.000 | yes |
| DeepSeek Distill D128 | 6,288.69 | 6,270.89 tok/s | 0.9972× | 112 | 1.000 | yes |

Loss relative differences are `0.0219%/0.0134%`, inside the numerical gate. Neither model
passes the declared 1.01 improvement policy; Qwen's two routes execute the same kernels and
its difference is treated as process noise, not a regression caused by routing.

## DeepSeek profile rebuttal

Across three executed steps:

```text
repeat forward calls 336 → 168
repeat backward calls 168 → 168
all dispatches       8,058 → 8,058
all Kernel time      261.730 → 263.477 ms
```

Every removed Value-repeat Kernel is replaced by an extra KV-group GEMM dispatch in P×V or
dP. P×V's isolated 1.60× win is offset by dP and library submission structure.

![Selective GQA Value broadcast discarded](../assets/selective-gqa-value-broadcast-discard.svg)

## Decision

Reject complete forward+dP routing and default false. The last distinct zero-stride
hypothesis is forward-only P×V: remove one Value expansion per layer while backward retains
the single H-batched dP. If it fails DeepSeek end to end, close the zero-stride family.

Raw evidence is in
[`benchmarks/results/2026-08-23-selective-gqa-value-broadcast/`](../../../benchmarks/results/2026-08-23-selective-gqa-value-broadcast/).
