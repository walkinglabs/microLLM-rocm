# Profiling and performance analysis

## Available today

### In-process trace API

`microllm::profiling::TraceSession`, `ScopedTraceSession`, and `TraceTimer` record
schema-versioned operator/layer/model traces. Value capture and timing are deliberately
separate passes so Tensor copies performed for diagnosis do not become benchmark time.

The complete cross-framework workflow is documented in [alignment.md](alignment.md).

### Micro-benchmark

```bash
MICROLLM_BUILD_DIR=build/hip-release \
MICROLLM_BENCH_DEVICE=hip \
./scripts/run_benchmarks.sh
```

The harness records warm-up, repetitions, HIP Event time, synchronized wall time,
numerical error, device metadata, and memory counters as JSON Lines.

### End-to-end benchmark

```bash
./build/hip-release/benchmarks/microllm_bench_model \
  --mode generate --model tiny --device hip \
  --warmup 2 --steps 10 --batch 1 --context 8 --new-tokens 32
```

### rocprofv3 trace

```bash
./scripts/profile_hip.sh /tmp/microllm-trace -- \
  ./build/hip-release/benchmarks/microllm_bench_model \
  --mode train --model tiny --device hip \
  --steps 5 --warmup 1 --batch 1 --context 8 --new-tokens 8
```

This produces HIP API/kernel/allocation traces, statistics, JSON, CSV, and Perfetto
output when supported by the installed rocprofv3.

## What remains

There is no Python `@profile` decorator yet. The current stable entry points are the
C++ RAII trace API and the alignment runner. Future work must correlate ranges with
rocprof markers, support asynchronous HIP Event completion without per-range device
synchronization, and export Perfetto ranges directly.

Ad-hoc wall-clock timing around asynchronous kernels is not accepted evidence.
