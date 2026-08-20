# Experiment 002 — transpose-aware GEMM

Status: `keep`

## Observed bottleneck

After Experiment 001, Qwen training strided-copy kernels occupy `33.55%` of Kernel
time. The inference baseline attributes `43.43%` to strided copies, dominated by a
roughly 544 MB tied-embedding transpose per cached forward.

Current autograd also materializes every Linear weight transpose during backward.

## Hypothesis

Submitting logical transpose flags directly to readable HIP and hipBLASLt GEMM will
remove full-weight copies, reduce allocation/peak memory, and improve both official
train rows plus Qwen tied-weight generation.

The hypothesis is weakened if vocabulary/weight-sized strided copies disappear but the
fixed workload score does not improve.

## Scope

- allowed: 2D transpose-aware matmul API/dispatch, HIP kernel indexing, hipBLASLt op
  flags/layouts, autograd gradient formulas, tied output call sites and focused tests;
- unchanged: allocator, dtype, Attention decomposition, KV Cache, model parameters,
  optimizer and benchmark protocol;
- no persistent duplicate tied weight in this experiment;
- no tolerance relaxation or global synchronization.

## Matrix contract

```text
C = op(A) × op(B)
op ∈ {identity, transpose}
```

Physical Tensor storage remains contiguous. Transpose is interpreted by the GEMM, not
materialized into a new Tensor.

## Required gates

- [x] CPU NN/NT/TN/TT hand values
- [x] readable HIP NN/NT/TN/TT
- [x] hipBLASLt NN/NT/TN/TT
- [x] FP32/FP16/BF16 HIP coverage with zero payload host transfer
- [x] autograd tied-head output and both gradients versus PyTorch
- [x] tied Qwen exact logits/tokens and multi-step loss trajectory
- [x] allocation/trace proves the full tied-weight copy is absent
- [x] full CPU/sanitizer/HIP/PyTorch regressions

## Implementation

- added a contiguous 2D `C = op(A) × op(B)` overload without creating a transpose Tensor;
- kept ordinary batched matmul and non-contiguous view behavior unchanged;
- added a readable typed HIP Kernel for NN/NT/TN/TT;
- mapped the same flags into the row-major-to-column-major hipBLASLt submission;
- expressed all four 2D autograd gradient formulas as transpose-aware GEMMs;
- changed tied training and cached inference output heads to reuse the embedding storage;
- did not create a permanent copied weight or add synchronization.

Descriptor/algorithm caching was intentionally deferred. It is a different variable and
must be measured separately.

## Correctness result

- CPU debug: `150/150` pass;
- ASan/UBSan: `148/148` pass;
- HIP release: `41/41` pass;
- Python/PyTorch operator parity: `4/4` pass;
- HIP NN/NT/TN/TT passes in FP32, FP16 and BF16;
- graph test proves the tied-head forward graph contains neither a `transpose` nor a
  `contiguous` node;
- Qwen and DeepSeek exact generated token lists are unchanged;
- Qwen first/final loss `0.217306778/0.000931422`; DeepSeek
  `9.687631607/8.080487251`; parameters change and optimizer payload transfers remain zero.

## Performance result

| Workload | Experiment 001 | Experiment 002 | Step speedup | PyTorch ratio | Peak memory ratio |
|---|---:|---:|---:|---:|---:|
| Qwen train | 24.027 token/s | 38.772 token/s | 1.61× | 0.755435 | 0.896769 |
| Qwen generate | 18.847 token/s | 35.355 token/s | 1.88× | 0.503757 | 0.960639 |
| DeepSeek train | 13.295 token/s | 22.356 token/s | 1.68× | 0.852426 | 0.797368 |
| DeepSeek generate | 10.053 token/s | 10.145 token/s | 1.01× | 0.162589 | 0.988725 |

```text
geometric score before  0.318328
geometric score after   0.479227
score improvement       50.5%
```

Qwen peak engine allocation falls from 9.56 GB to 9.01 GB in training and from
2.52 GB to 1.98 GB in inference. Total allocated bytes fall by about 45.0% in Qwen
training and 99.3% in Qwen inference.

## Profiler result

Representative rocprof Qwen train (`1` warm-up + `1` measured):

```text
strided-copy calls             1302 → 624       (-52.1%)
strided-copy Kernel time   62.327 ms → 2.156 ms (-96.5%)
all Kernel time           185.764 ms → 117.944 ms
hipMalloc calls                5423 → 4745
ordinary Kernel launches       4698 → 4020
profiled measured step      about 149.1 ms → 99.8 ms
```

The remaining 624 short strided copies are ordinary view materializations elsewhere;
the vocabulary/weight-sized copies are gone. RMSNorm forward/backward is now 64.31% of
Kernel time, so Step 03 has a measured reason to exist.

Raw JSONL and compact profiler tables are in [002-data](002-data/README.md). The large
PFTrace remains at `/tmp/microllm-qwen-train-profile-exp002/` on the measurement host.

## Commands

```bash
python3 benchmarks/single_gpu/hf_model_matrix.py \
  --manifest build/hip-release/benchmarks/hf-models.local.json \
  --infer-binary build/hip-release/apps/microllm_hf_infer \
  --train-binary build/hip-release/apps/microllm_hf_train_step \
  --device hip --modes infer,train \
  --output /tmp/microllm-exp002/microllm.jsonl
```

## Decision

`keep`.

The main hypothesis is supported. It also produced a useful negative result: DeepSeek
generation barely changes because this selected model path does not pay the same tied
output transpose cost. The next single variable is block-parallel RMSNorm.
