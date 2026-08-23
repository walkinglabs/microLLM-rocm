# Experiment 164 — keep Value and context in BTHD through the complete graph

## Boundary

After Q/K RoPE layout fusion, four materializations remained per layer/step:

1. Value BTHD→BHTD before Attention;
2. context BHTD→BTHD before output projection;
3. output-projection gradient BTHD→BHTD before Attention backward;
4. Value gradient BHTD→BTHD before projection backward.

Experiment 163 proved P×V can use interleaved BTHD. This experiment adds matching
`dP=dO×Vᵀ` and `dV=Pᵀ×dO` hipBLASLt layouts, then changes the causal GQA graph contract:

```text
Q [B,H,T,D]      K [B,KV,T,D]
V [B,T,KV,D]  →  Attention  →  context [B,T,H,D]
```

GQA head expansion/reduction moves from dimension 1 to dimension 2 for Value only. Q/K
math, causal softmax and scale are unchanged.

## Correctness proof

- CPU dP/dV and full GQA forward/backward match the materialized reference;
- graph test compares output and Q/K/V gradients and proves the fused graph has no
  transpose/contiguous node around Attention;
- Python reconstructs the BTHD expression with PyTorch and compares output, dP, expanded
  dV and reduced Q/K/V graph gradients;
- HIP B2 primitive tests cover interleaved batches with zero payload transfer;
- HIP T256 saved forward/backward compares every probability, output and Q/K/V gradient;
- complete CPU/HIP Transformer compares loss and every named parameter gradient.

## Same-binary official T512 A/B

RoPE layout fusion stays enabled. Only `--attention-context-layout-fusion` changes. Each
model/policy uses three fresh processes, B1, one warm-up and two measured BF16-Linear/FP32-
master steps.

| Model | Materialized | Fused | Speedup | Peak saved | Allocations saved |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 14,813.50 | 15,311.55 tok/s | 1.0336× | 100,401,152 B | 192 |
| DeepSeek Distill 1.5B | 6,170.24 | 6,328.01 tok/s | 1.0256× | 205,520,896 B | 224 |

Qwen final-loss relative difference is `0.00694%`; DeepSeek is equal at reported
precision. The observed parameter guard is equal for both.

Diagnostics now report:

```text
Qwen     96 calls / 100,663,296 B → 0 / 0
DeepSeek 112 calls / 205,520,896 B → 0 / 0
Autograd materializations          → 0
```

The peak bytes saved closely match one model step's remaining diagnosed layout bytes.

rocprofv3 on the same Qwen workload reports dispatches `7,192→6,907`, strided-copy
`288→0`, and total Kernel time `111.727→110.668 ms`. The uninstrumented fresh-process
medians, not profiler throughput, decide the speed gate.

![Complete Attention context layout](../assets/attention-context-layout-fusion.svg)

## Decision

Keep and enable by default. The candidate removes every strided-copy layout diagnosed in
Experiment 161, reduces peak on both official models, improves both throughput medians and
passes all independent gradients. The explicit false policy remains for same-binary
rebuttal and unsupported short paths retain the materialized operator fallback.

Raw evidence is in
[`benchmarks/results/2026-08-23-attention-context-layout-fusion/`](../../../benchmarks/results/2026-08-23-attention-context-layout-fusion/).
