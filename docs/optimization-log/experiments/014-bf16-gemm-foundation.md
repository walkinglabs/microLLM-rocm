# Experiment 014 — BF16 mixed-GEMM foundation

Status: `keep` — foundation only, not whole-model BF16

## Track boundary

This starts a separate BF16 track. It does not alter the FP32 running-best curve and
does not claim whole-model BF16. The node must first prove device-native cast and
BF16-input/FP32-output GEMM correctness and speed on MI300X.

## Hypothesis

MI300X BF16 matrix hardware can accelerate representative Linear shapes while FP32
accumulation/output keeps a simple master-weight/autograd boundary. Native cast must not
round-trip through the host.

## Scope

- FP32/FP16/BF16 device cast;
- `left FP32 → BF16`, cached/right BF16, hipBLASLt compute FP32, output FP32;
- CPU rounded reference;
- no model policy, optimizer, loss scaling or full-network claim yet.

## Required gates

- [x] cast round-trip, special values and zero host transfer
- [x] rectangular BF16 mixed GEMM versus rounded CPU/PyTorch reference
- [x] representative Model-S/Qwen/DeepSeek Linear shapes
- [x] FP32 versus BF16 Kernel timing and error
- [x] full regressions

## Implementation

- generic device cast for FP32/FP16/BF16 input/output combinations;
- no Tensor payload passes through CPU;
- mixed GEMM accepts FP32 activation plus pre-cast BF16 weight;
- activation casts to BF16 on device;
- hipBLASLt multiplies BF16 inputs with FP32 compute and FP32 output;
- CPU path explicitly rounds both operands before FP32 reference matmul;
- no model precision policy or cached BF16 model weights in this node.

## Correctness result

- CPU debug: `153/153` pass;
- ASan/UBSan: `151/151` pass;
- HIP release: `55/55` pass;
- Python/PyTorch operator parity: `4/4` pass;
- Inf/-Inf/NaN and signed zero survive native BF16 round trip;
- 128×128 mixed GEMM matches the independently rounded CPU reference;
- focused cast/GEMM performs zero H2D/D2H during execution.

## MI300X M=1 shape benchmark

Each row includes activation cast, 10 warm-ups and 50 Event-timed repetitions:

| Shape M×K×N | FP32 ms | BF16 mixed ms | Speedup | Max error |
|---|---:|---:|---:|---:|
| 1×384×384 | 0.02536 | 0.02928 | 0.87× | 1.19e-7 |
| 1×896×896 | 0.03074 | 0.02877 | 1.07× | 6.0e-8 |
| 1×896×4864 | 0.02876 | 0.03468 | 0.83× | 6.0e-8 |
| 1×1536×1536 | 0.02986 | 0.03129 | 0.95× | 2.38e-7 |
| 1×1536×8960 | 0.04026 | 0.03509 | 1.15× | 4.17e-7 |

BF16 is not universally faster. A whole-model policy must be per shape and cache BF16
weights; blindly converting every Linear would regress Model-S and Qwen FFN shapes.

Raw JSONL is in [014-data](014-data/README.md), and the generated chart is
[bf16-gemm.svg](../assets/bf16-gemm.svg).

## Results

Supported as an operator foundation. Falsified as a universal M=1 replacement.

## Decision

`keep` for native cast/mixed GEMM APIs and tests. Whole-model BF16 remains incomplete.
