# Experiment 003 — block-parallel RMSNorm

Status: `keep`

## Observed bottleneck

After Experiment 002, RMSNorm forward/backward consumes `64.31%` of Qwen training
Kernel time. Each old GPU thread loops through a complete hidden row.

## Hypothesis

One block per row, parallel FP32 reductions and a non-atomic weight-gradient reduction
will remove RMSNorm as the dominant Kernel group and improve all fixed workloads.

The hypothesis is weakened if RMSNorm Kernel time falls sharply while fixed end-to-end
throughput does not improve.

## Scope

- allowed: FP32 HIP RMSNorm forward/backward kernels, device scratch and focused tests;
- unchanged: CPU formula, public API, dtype, model graph, GEMM, allocator, KV Cache and
  benchmark protocol;
- no tolerance relaxation, atomic weight-gradient hot spot or global synchronization.

## Required gates

- [x] widths 16/384/512/896/1536 and rows 1/3/32
- [x] zeros, mixed-sign values, large values and epsilon variants
- [x] forward/backward CPU/HIP
- [x] PyTorch oracle and finite differences
- [x] zero Tensor-payload host transfer
- [x] official multi-step loss/tokens/parameter update
- [x] full CPU/sanitizer/HIP regressions

## Implementation

- forward assigns one 256-thread block to each row;
- every thread reads multiple columns and the block performs an FP32 square-sum reduction;
- backward reduces both square sum and weighted dot in parallel;
- backward stores one inverse RMS value per row in device scratch;
- a second Kernel assigns one thread to each weight column and reduces rows without
  atomics;
- output values and public RMSNorm/autograd contracts are unchanged.

## Correctness result

- CPU debug: `150/150` pass;
- ASan/UBSan: `148/148` pass;
- HIP release: `42/42` pass;
- Python/PyTorch operator parity: `4/4` pass;
- focused HIP test covers 15 rows/width combinations and two real epsilon values;
- focused forward/backward performs zero Tensor-payload H2D/D2H;
- exact Qwen and DeepSeek generated token lists remain unchanged;
- final measured loss remains `0.000931422` for Qwen and `8.080489159` for DeepSeek;
- the small early-logit/loss delta is within the existing FP32 gate and comes from the
  now-parallel reduction order.

## Performance result

| Workload | Experiment 002 | Experiment 003 | Step speedup | PyTorch ratio | Peak memory ratio |
|---|---:|---:|---:|---:|---:|
| Qwen train | 38.772 token/s | 71.057 token/s | 1.83× | 1.384483 | 0.896769 |
| Qwen generate | 35.355 token/s | 57.322 token/s | 1.62× | 0.816751 | 0.960639 |
| DeepSeek train | 22.356 token/s | 47.913 token/s | 2.14× | 1.826933 | 0.797368 |
| DeepSeek generate | 10.145 token/s | 18.597 token/s | 1.83× | 0.298040 | 0.988725 |

```text
geometric score before  0.479227
geometric score after   0.885816
score improvement       84.8%
```

The two training rows exceed the fixed PyTorch ratio of `1.0`, but this workload has
batch 1, only three predicted tokens per step and FP32. It is evidence for this selected
matrix, not a claim about long-context or production training.

## Profiler result

Representative Qwen train rocprof (`1` warm-up + `1` measured):

```text
RMSNorm forward/backward time   75.849 ms → 1.554 ms (-98.0%)
RMSNorm Kernel share                64.31% → 3.59%
all Kernel time                 117.944 ms → 43.254 ms
profiled measured step            99.8 ms → 57.6 ms
```

The new largest single Kernel group is device-native AdamW at 25.02% of Kernel time,
but generation remains the weaker side of the score. The next planned experiment moves
to the KV/GQA data path rather than automatically optimizing the largest training share.

Raw JSONL and compact profiler tables are in [003-data](003-data/README.md). The large
PFTrace remains at `/tmp/microllm-qwen-train-profile-exp003/` on this host.

## Commands

```bash
python3 benchmarks/single_gpu/hf_model_matrix.py \
  --manifest build/hip-release/benchmarks/hf-models.local.json \
  --infer-binary build/hip-release/apps/microllm_hf_infer \
  --train-binary build/hip-release/apps/microllm_hf_train_step \
  --device hip --modes infer,train \
  --output /tmp/microllm-exp003/microllm.jsonl
```

## Results

The main hypothesis is supported.

## Decision

`keep`. The next single variable is the device-resident, preallocated KV/GQA path in
Experiment 004.
