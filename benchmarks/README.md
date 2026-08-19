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

`microllm_bench_model` measures train or cache-backed generation throughput. Its
`tokens_per_second` excludes construction and warm-up; `tokens_per_second_with_setup`
includes construction, device transfer, optimizer allocation, and warm-up. Both are
reported so setup cannot disappear from the experiment.

Capture HIP API, kernel, memory, and RCCL-ready runtime traces with the locally
installed rocprofv3 interface:

```bash
./scripts/profile_hip.sh /tmp/microllm-trace -- \
  ./build/benchmarks/microllm_bench_model \
  --mode train --model tiny --device hip --steps 5 --warmup 1 \
  --batch 1 --context 8 --new-tokens 8
```
