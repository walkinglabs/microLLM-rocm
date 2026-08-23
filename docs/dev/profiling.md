# Profiling and performance analysis

## Available today

### In-process trace API

`microllm::profiling::TraceSession`, `ScopedTraceSession`, and `TraceTimer` record
schema-versioned operator/layer/model traces. Value capture and timing are deliberately
separate passes so Tensor copies performed for diagnosis do not become benchmark time.

The complete cross-framework workflow is documented in [alignment.md](alignment.md).

Official-model value diagnosis can opt into every Transformer block's detail records:

```bash
./build/hip-release/apps/microllm_hf_infer \
  --config model/config.json --weights model/model.safetensors \
  --tokens 1,2,3,4,5,6,7,8 --device hip --top-k 1 --new-tokens 0 \
  --warmup 0 --steps 1 --prefill-warmup 0 --prefill-steps 1 \
  --prefill-logits last --workload prefill --use-cache true \
  --kv-cache-dtype fp32 --trace-output /tmp/all-layers.jsonl \
  --trace-max-elements 1 --trace-all-layer-details true
```

`trace-max-elements=1`只限制JSON中保存的样例值；min/max/mean/L2统计仍扫描完整Tensor。
全层trace会同步并把诊断值带回CPU，所以只能回答数值范围问题，不能作为性能数据。默认仍只记录
block 0细节，旧trace的数量和名字不会改变。

### Micro-benchmark

```bash
MICROLLM_BUILD_DIR=build/hip-release \
MICROLLM_BENCH_DEVICE=hip \
./scripts/run_benchmarks.sh
```

The harness records warm-up, repetitions, HIP Event time, synchronized wall time,
numerical error, device metadata, and memory counters as JSON Lines.

AdamW has its own implementation-selectable benchmark:

```bash
./build/hip-release/benchmarks/microllm_bench_adamw \
  --elements 802816 --mirror true \
  --implementation scalar --warmup 5 --repetitions 20

./build/hip-release/benchmarks/microllm_bench_adamw \
  --elements 802816 --mirror true \
  --implementation vectorized --warmup 5 --repetitions 20
```

It reports HIP Event time, effective bytes/s and a sampled numerical guard. `Auto` remains
the validated scalar policy; an explicit candidate result is not a default dispatch claim.

`microllm_bench_ops` 的 matmul 路径还接受 `--batch`，例如：

```bash
./build/hip-release/benchmarks/microllm_bench_ops \
  --op matmul --device hip --implementation hipblaslt \
  --batch 14 --m 512 --k 64 --n 512 --transpose-right true \
  --warmup 3 --repetitions 10
```

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
C++ RAII trace API, operator micro-benchmarks and the alignment runner. Future work must correlate ranges with
rocprof markers, support asynchronous HIP Event completion without per-range device
synchronization, and export Perfetto ranges directly.

Ad-hoc wall-clock timing around asynchronous kernels is not accepted evidence.
