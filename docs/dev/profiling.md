# Profiling and performance analysis

## Available today

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

## What is not implemented

There is no in-process `@profile` decorator or stable C++ profiling session API yet.
The intended C++ boundary is an RAII `ProfileSession`/`ProfileRange`, followed by an
optional Python context manager/decorator. It must correlate ranges with Stream/Event
timing and rocprof markers without forcing a global synchronization.

Until that API exists, performance work must use the benchmark executables and
`scripts/profile_hip.sh`; ad-hoc wall-clock timing around asynchronous kernels is not
accepted evidence.
