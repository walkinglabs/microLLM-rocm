# Experiment 196 — bind grouped gate/up to the FFN Arena

Status: keep explicit policy; default unchanged

## Design

The model may use grouped gate/up only through bf16_ffn_out_, because that path owns stable Arena
input, gate and up addresses. The exact registry key contains rows, inner, columns, architecture,
HIP runtime/driver and hipBLASLt version.

One initialized GroupedGemm kernel is cached by shape, index, device and Stream. A much smaller
plan key adds input/output plus gate/up weight pointers. Every block creates one device argument
record; later forwards execute only run. Unregistered shapes fall through to the old two GEMMs.

## Complete-model gate

| Model | Baseline | Grouped | Speedup | Max/RMS | Peak | Setup |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 93471 tok/s | 95118 tok/s | 1.0176× | 0.07028/0.01538 | 1.000008× | 57.0 ms |
| DeepSeek | 50157 tok/s | 50746 tok/s | 1.0117× | 0.06139/0.01029 | 1.000003× | 56.8 ms |

All 12 complete outputs are finite, preserve top-1 and pass the previously declared BF16
0.25/0.05 boundary. Qwen has 24 plans and DeepSeek 28; after two warm-ups plus five measured
forwards their plan hits are exactly 144/168.

![Grouped gate/up model gate](../assets/bf16-grouped-gate-up-model.svg)

## Phase-delta evidence

Qwen GEMM calls fall 217→193 and GEMM time improves 1.035×; total Kernel improves 1.018×.
DeepSeek calls fall 253→225 and GEMM time improves 1.020×, while instrumented total Kernel is
0.998×. That profiler counterexample is retained instead of being hidden behind the
uninstrumented throughput result.

## Decision

Keep the registry, shared initialized kernel, per-block device arguments, CLI counters and
explicit T512 policy. Do not install 65168/65200 as defaults: solution indices are backend-local,
short/batch shapes remain unmeasured, and the policy requires FFN Arena address ownership.

Raw evidence:
[benchmarks/results/2026-08-24-bf16-grouped-gate-up-model/](../../../benchmarks/results/2026-08-24-bf16-grouped-gate-up-model/).
