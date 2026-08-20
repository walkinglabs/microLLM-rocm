# MI300X single-GPU built-in model matrix

This directory retains the first six-row `Benchmark.HipModelMatrix` run: tiny,
Model-S, and Model-M, each in train and generate mode.

```text
date          2026-08-20 UTC
device        AMD Instinct MI300X VF
architecture  gfx942:sramecc+:xnack-
dtype         FP32
CTest result  pass, 11.30 seconds
```

The configurations are intentionally small enough for a regular HIP gate. They are
not training-quality or steady-state serving benchmarks. `device_peak_engine_bytes`
tracks engine-owned allocations only.

Reproduce with:

```bash
ctest --test-dir build/hip-release \
  -R '^Benchmark.HipModelMatrix$' \
  --output-on-failure -V
```
