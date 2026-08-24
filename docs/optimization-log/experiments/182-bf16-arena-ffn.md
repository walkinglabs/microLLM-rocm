# Experiment 182 — caller-owned BF16 FFN

Status: `keep` workspace and shape-selective Graph candidate; model gate required

## Question

Experiment 181 used FP32. Production inference already uses BF16 FFN weights and
intermediates, and some gfx942 shapes cannot write FP32 output directly. Can one
caller-owned contract remove allocations without dropping those shapes?

## Contract

`Bf16FfnWorkspace` owns five caller-provided BF16 buffers:

```text
input cast → gate → up → SwiGLU → optional down fallback
```

`bf16_matmul_output_out_` first attempts direct BF16×BF16→FP32. When the installed
hipBLASLt rejects that exact shape, it writes BF16 into the caller fallback and casts
to the caller FP32 output. No hidden Tensor is allowed in either route.

CPU/HIP tests cover output parity, dtype/shape/device/layout, aliases, zero payload
transfers, and the Qwen R1 fallback.

## Formal matrix

Three fresh processes per policy; each excludes three warm-ups and times twenty
regions:

| Workload | Arena eager | Arena Graph | Nodes | Setup break-even |
|---|---:|---:|---:|---:|
| Qwen R1 | 1.063× | 1.182× | 6 | 20 |
| Qwen R32 | 1.065× | 1.083× | 6 | 11 |
| Qwen R512 | 5.548× | 5.049× | 5 | 1 |
| DeepSeek R1 | 1.038× | 1.068× | 6 | 37 |
| DeepSeek R32 | 1.064× | 0.970× | 5 | never |
| DeepSeek R512 | 4.057× | 3.837× | 5 | 1 |

All 54 complete outputs are bit-exact. Arena and Graph each pass five of six 1.05
rows. DeepSeek R32 is the counterexample that prevents universal Graph routing.

![BF16 Arena FFN result](../assets/bf16-arena-ffn.svg)

## Allocation and profiler evidence

The timed allocation counters report 100–120 allocation/deallocation calls for
twenty baseline regions and zero for Arena/Graph. In the Qwen R512 whole-process
rocprofv3 profile:

| Counter | Baseline | Arena | Arena Graph |
|---|---:|---:|---:|
| Executed Kernels | 130 | 130 | 130 |
| Kernel duration | 1.79 ms | 1.59 ms | 1.54 ms |
| malloc/free | 127 / 126 | 12 / 11 | 12 / 11 |
| direct launch APIs | 129 | 129 | 19 |
| Graph launches | 0 | 0 | 23 |

The speedup is allocation/submission removal. The algebra and Kernel count are not
reduced.

## Decision

Keep the out APIs, workspace, fallback, benchmark and evidence. Do not enable the
Graph in `TransformerModel`: input/output addresses are not stable across the full
model, and DeepSeek R32 is slower. The next node must route the eager Arena through
complete BF16 Qwen/DeepSeek logits and measure end-to-end inference before considering
any model default.

Raw evidence:
[`benchmarks/results/2026-08-24-bf16-arena-ffn/`](../../../benchmarks/results/2026-08-24-bf16-arena-ffn/).
