# Experiment 181 — first heterogeneous Transformer region in the arena

Status: `keep` shape-selective FP32 FFN foundation; not a model default

## Why this is different

Experiment 180 captured an add chain. This node captures the actual dense FFN algebra and official
matrix shapes:

```text
gate = X × W_gate
up   = X × W_up
activated = SwiGLU(gate, up)
output = activated × W_down
```

The region mixes three hipBLASLt GEMMs with a repository SwiGLU Kernel.

## Required interfaces

`Storage::from_external` creates an explicit non-owning Storage descriptor; the caller owns
lifetime and synchronization. `swiglu_out_` writes into caller Storage. Combined with existing
`matmul_out_`, Arena slices can be wrapped as non-owning Tensors without changing addresses.

CPU/HIP tests cover external lifetime, mutation, null rejection, out-shape/dtype/contiguity and
the existing low-precision family. The Arena continues to own and synchronize its backing memory.

## Formal matrix

FP32 official dimensions, three fresh processes per mode:

| Workload | Arena eager | Arena Graph | Setup | Break-even |
|---|---:|---:|---:|---:|
| Qwen R32 | 1.148× | 1.202× | 0.286 ms | 23 |
| Qwen R512 | 3.033× | 2.970× | 0.266 ms | 1 |
| DeepSeek R32 | 1.042× | 1.005× | 0.257 ms | 568 |
| DeepSeek R512 | 1.667× | 1.679× | 0.278 ms | 2 |

All 36 processes have bit-exact complete output. Graph node count is exactly four. Arena backing
is 1.87/29.88 MB for Qwen and 3.44/55.05 MB for DeepSeek at R32/R512.

![Arena FFN result](../assets/arena-ffn.svg)

## Profiler

For Qwen R512:

| Counter | Deferred | Arena | Arena Graph |
|---|---:|---:|---:|
| Executed Kernels | 101 | 101 | 101 |
| Kernel duration | 3.23 ms | 3.10 ms | 3.12 ms |
| malloc/free | 80 / 79 | 11 / 10 | 11 / 10 |
| direct host Kernel launches | 100 | 100 | 12 |
| Graph launches | 0 | 0 | 23 |

The improvement is allocator/host submission, not changed math.

## Decision

Keep external Storage, `swiglu_out_`, the FFN benchmark and shape-selective arena/Graph candidate.
Do not enable it in the model: DeepSeek R32 is a counterexample, and current production inference
uses BF16 FFN weights. The next necessary node is caller-owned BF16-output/FP32-output GEMM plus a
complete model FFN routing gate.

Raw evidence is in
[`benchmarks/results/2026-08-24-arena-ffn/`](../../../benchmarks/results/2026-08-24-arena-ffn/).
