# Reproducible benchmarks

Build in Release mode. Every micro benchmark records warm-up count, repetitions,
device metadata, HIP runtime/driver versions, kernel Event time, synchronized wall
time, numerical error against the CPU reference, and before/after device memory.

```bash
./scripts/configure.sh -DCMAKE_BUILD_TYPE=Release -DMICROLLM_ENABLE_HIP=ON
MICROLLM_BENCH_DEVICE=hip ./scripts/run_benchmarks.sh
```

The output is JSON Lines. `kernel_ms_min` is never used alone as the project-level
performance claim. Before/after free memory is diagnostic and is not peak memory.
Peak memory must come from a dedicated allocator tracker or profiling tool.

Result schema version 1 fields are emitted directly by `microllm_bench_ops`.
Representative committed smoke results live under `benchmarks/results/`; full local
run outputs are ignored unless curated with their environment and correctness data.
