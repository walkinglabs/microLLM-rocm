# Experiment 025 — paired K/V cache store

Status: `keep`

K and V were written by two identical Kernels at the same layer and position. The new
operator validates both cache contracts and writes both arrays in one launch. CPU keeps
the readable two-store reference.

Correctness: CPU `157/157`, sanitizer `155/155`, HIP `58/58`; a focused test compares
both caches and proves zero payload transfer during execution. Full cached GQA and
official tokens remain exact.

Three-process inference medians:

```text
Qwen       218.40 → 219.30 token/s  +0.4%
DeepSeek    78.10 →  78.74 token/s  +0.8%
score       2.470863 → 2.478439
```

The gain is small, so it is accepted only because the prospective three-process rule is
satisfied, neither workload regresses, one launch per layer is removed, and the public
pair operator has direct tests. Raw evidence is in [025-data](025-data/README.md).
