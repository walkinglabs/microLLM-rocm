# rocWMMA BF16 QK tile matrix

This directory contains the first bounded matrix-core capability gate for a future
online Attention implementation. It does not contain a model-speed claim.

## Reproduce

Build on a HIP system with rocWMMA and hipBLASLt available, then run:

```bash
cmake --preset hip-release
cmake --build build/hip-release --target microllm_bench_rocwmma_qk --parallel
ROCR_VISIBLE_DEVICES=0 python3 benchmarks/single_gpu/rocwmma_qk_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_rocwmma_qk \
  --output-directory benchmarks/results/2026-08-25-rocwmma-qk-tile \
  --runs 3 --warmup 10 --repetitions 50
```

Environment: AMD Instinct MI300X VF (`gfx942:sramecc+:xnack-`), ROCm 7.13,
rocWMMA 2.2.0. The matrix contains T16–T2048 and head widths 64/128. Each of the
16 shapes runs in three fresh processes.

## Files

- `raw.jsonl`: 48 complete per-process records;
- `summary.json`: per-shape process medians and admission decision;
- `verification.json`: machine-readable counts and safety gates.

Every process compares all `T×T` outputs against a CPU BF16-rounded reference and
also times a scalar HIP kernel and hipBLASLt with the same semantic input and
caller-owned FP32 output.

The selected capability layout is 32×32×16 with one wave per block; T16 uses
16×16×16. It is 1.784×/1.654× the hipBLASLt baseline at T512 D64/D128, but only
0.688× at T2048 D128. This counterexample is why the decision is only to admit a
bounded online-Attention prototype. Causal masking, GQA, tails, online softmax,
PV accumulation, memory and complete-model logits remain unimplemented gates.
