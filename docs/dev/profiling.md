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
  --trace-max-elements 1 --trace-all-layer-details true \
  --trace-value-filter attention_norm,attention.context,ffn_norm,ffn.activated
```

`trace-max-elements=1`只限制JSON中保存的样例值；min/max/mean/L2统计仍扫描完整Tensor。
`trace-value-filter`让未匹配记录只保留名字、shape和numel，避免为了一个问题搬运所有中间值。
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

The legacy comparison reports HIP Event time, effective bytes/s and a sampled numerical
guard. Use the correctness-first tuner for a dispatch decision:

```bash
./build/hip-release/benchmarks/microllm_tune_adamw \
  --elements 802816 --mirror true --aligned true \
  --warmup 3 --repetitions 20 --mode training --accept false
```

It compares every parameter, first moment, second moment and optional BF16 mirror value
before timing. Its exact key also isolates alignment, mirror, architecture, HIP versions
and mode. Screening does not mutate `Auto`; explicit acceptance and cache persistence must
follow a separate end-to-end regression. The current MI300 matrix keeps Scalar as fallback.

### Exact operator implementation registries

After a candidate passes the numerical and repeated-timing gates, register it with a key
made from the real operands and execution context:

```cpp
microllm::ops::OpContext context;
context.mode = microllm::ops::OpMode::Inference;
context.workspace_bytes = 4 * 1024 * 1024;
const auto key = microllm::ops::make_matmul_tuning_key(
    left, right, false, false, context);
microllm::ops::register_matmul_implementation(
    key, microllm::ops::MatmulImplementation::HipBLASLt);
```

The key includes dtype, transpose/layout/strides, GPU architecture, HIP runtime/driver,
hipBLASLt version, mode and workspace limit in addition to M/K/N. A choice measured for
FP32 NN cannot leak into FP16, TT, training, another workspace budget or another software
stack. Registration itself does not benchmark or prove correctness; the micro-benchmark
and end-to-end regression remain required. A validated registry can be persisted and restored:

```cpp
microllm::ops::save_matmul_tuning_cache("matmul-cache.jsonl");
const auto report = microllm::ops::load_matmul_tuning_cache(
    "matmul-cache.jsonl", microllm::Device::hip(0));
```

The schema-versioned JSONL loader is transactional. Architecture, HIP runtime/driver and
hipBLASLt mismatches are reported as stale and never activated. Persistence does not turn
registration into an autotuner; correctness-before-timing is still required.

The matmul harness now implements that missing order:

```bash
./build/hip-release/benchmarks/microllm_tune_matmul \
  --m 128 --k 128 --n 128 --dtype fp32 \
  --warmup 3 --repetitions 10 --mode inference --accept false
```

Every candidate first compares its complete output with Readable. A failed candidate has
zero timing fields. Passing candidates receive default-Stream HIP Event and synchronized
wall P50/P95. The command only screens by default; `--accept true --cache-output ...` is
explicit and still requires a separate model end-to-end regression.

BF16 Linear training can also screen version-local hipBLASLt solution indices:

```bash
./build/hip-release/benchmarks/microllm_tune_bf16_algorithms \
  --rows 512 --inner 896 --columns 896 --output-dtype fp32 \
  --maximum-algorithms 64 --workspace-bytes 33554432 \
  --warmup 2 --repetitions 5
```

It clears its process-local registry after screening and never writes a default. The
explicit `hf_train_step --bf16-algorithms M:K:N:index,...` seam exists for model rebuttal
experiments; solution indices are backend-version local and are not a persistent policy.

`microllm_bench_ops` 的 matmul 路径还接受 `--batch`，例如：

```bash
./build/hip-release/benchmarks/microllm_bench_ops \
  --op matmul --device hip --implementation hipblaslt \
  --batch 14 --m 512 --k 64 --n 512 --transpose-right true \
  --warmup 3 --repetitions 10
```

### Autograd and strided-layout attribution

`microllm_hf_train_step --diagnostics-output /tmp/diagnostics.json` enables two
thread-local metadata counters only for measured steps. Autograd records target
operation/shape, first contribution source, later add source and materialization;
Runtime records exact strided-copy shape/stride, calls, elements and bytes. Values are
not copied to host. The diagnostic bookkeeping is intentionally excluded from performance
measurements and is disabled by default.

The retained tied-weight optimization can be rebutted with the same binary using
`--tied-embedding-sparse-add true/false`.

The retained Q/K RoPE layout optimization has the matching same-binary control:
`--attention-rope-layout-fusion true/false`. `true` reads projection output in
`[B,T,H,D]` and writes RoPE output in `[B,H,T,D]`; `false` restores the explicit
transpose materializations. Diagnostics and timing must still be separate processes.

The remaining Value/context boundary uses
`--attention-context-layout-fusion true/false`. Keep
`--attention-rope-layout-fusion true` in both processes when isolating it. The true route
uses BTHD P×V/dP/dV layouts; false restores Value/context transpose nodes.

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
