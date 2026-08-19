# 2026-08-19 — M5 hipBLASLt candidate and shape-aware selection

## Contract

Keep readable matmul as reference. Add a 2D contiguous FP32 hipBLASLt candidate behind
an explicit implementation selector, require CPU numerical comparison, measure real
Model-S projection shapes, and report setup-inclusive regressions.

## Integration

- row-major `C=A×B` is described as column-major `Cᵀ=Bᵀ×Aᵀ` without data transpose;
- hipBLASLt handle is reused; operation/layout descriptors remain per call;
- caller Stream/workspace are propagated;
- `Auto` selects hipBLASLt for valid HIP 2D operations with K and N at least 128;
- batched matrices and small widths retain the readable kernel;
- Model Linear forward and Autograd matmul use Auto; batched Attention remains readable.

The first build failed because hipBLASLt public headers require a HIP platform macro
while `roc::hipblaslt` alone did not provide it to a C++ source. Linking `hip::host`
plus `roc::hipblaslt` fixed the host API boundary without propagating device offload
flags to g++.

## Shape evidence

| shape | readable mean | hipBLASLt mean | outcome |
|---|---:|---:|---|
| 64×64×64 | 0.05794 ms | 0.06142 ms | library slightly slower |
| M=1,K=384,N=384 | 0.24491 ms | 0.05496 ms | library 4.46× faster |

The first Auto heuristic used the smallest M/K/N dimension and incorrectly chose
readable for Model-S token projection because M=1. Measuring the real shape corrected
the policy to use K/N width.

## Model-S result and counterexample

Measured one-token generation:

- CPU reference: 9.33 tokens/s;
- readable HIP: 55.86 tokens/s;
- Auto hipBLASLt HIP: 187.10 tokens/s.

The optimized measured region is about 3.35× faster than readable HIP. However,
setup-inclusive throughput falls from 9.13 to 2.39 tokens/s because library/runtime
initialization dominates this tiny five-token experiment. Thus the same optimization
supports a steady-state speedup and exposes a startup regression. Both results are
retained; neither is replaced by the better-looking number.

Raw JSONL is committed under `benchmarks/results/2026-08-19-hipblaslt.jsonl`.
