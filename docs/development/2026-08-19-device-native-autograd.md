# 2026-08-19 — device-native HIP autograd graph

## Problem found by source audit

The eager reverse-mode graph engine and forward HIP kernels were repository-owned, but
several nonlinear backward closures called `Tensor::to_vector()`. That made the result
correct while silently executing Embedding, Softmax, RMSNorm, SwiGLU, RoPE, causal
Softmax, GQA repeat, and cross-entropy gradients on the CPU.

## Change

- kept the repository-owned `Value::Node` graph, parent edges, reverse-topological walk,
  branch accumulation, and repeated-backward semantics;
- added CPU reference plus readable HIP backward operators for reduction/broadcast,
  Embedding scatter-add, Softmax, RMSNorm, SiLU/SwiGLU, RoPE, cross entropy, causal
  Softmax, and repeat-interleave;
- rewired autograd closures to call Tensor operators instead of host vectors;
- added runtime transfer counters so a test can reject hidden host/device copies;
- retained the readable self-written matmul kernel for tiny shapes; optimized shapes may
  still explicitly select hipBLASLt through the existing implementation policy.

## Evidence

Commands:

```bash
cmake --build build-pa -j2
ctest --test-dir build-pa --output-on-failure -L cpu
cmake --build build-hipblaslt -j2
ctest --test-dir build-hipblaslt --output-on-failure -L hip
```

Results on the current MI300X/gfx942 environment:

- CPU reference and graph regression: 90/90 passed;
- HIP suite: 22/22 passed;
- device-native primitive backward comparison: passed;
- full one-layer GQA Transformer CPU/HIP loss and every-parameter gradient comparison:
  passed;
- host-to-device calls during GPU forward/backward graph: 0;
- device-to-host calls during GPU forward/backward graph: 0.

## Honest boundary

The forward/backward graph is now device-native for the implemented FP32 Transformer
path. A whole training step is not yet transfer-free: metrics and the current SGD/AdamW
implementation still read values on the host. The readable reduction and scatter kernels
are correctness baselines, not current performance claims.
