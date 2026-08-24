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

Dense accumulation additionally reports exclusive-owner candidate/executed calls and
elements. `--unique-gradient-inplace-add true/false` is the matching same-binary control.
It defaults false after Experiment 172: logical engine allocations fell, while backend/HIP
allocation calls, add launches, peak memory and two-model throughput did not clear the gate.

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

`--attention-layout-plan-cache true/false` controls the exact immutable descriptor/layout
cache. Its default is false after the Experiment 166 model rejection. The public
`attention_layout_plan_cache_stats()` and `clear_attention_layout_plan_cache()` APIs expose
entries/hits/misses without copying Tensor payloads.

`--attention-gemm-scale-fusion true/false` controls the QK/dQ/dK alpha experiment. Its
default is false after Experiment 167. The generic scaled-matmul operator remains available;
explicit Attention routing should be used only for numerical/performance rebuttals.

`--attention-paired-gqa-repeat true/false` selects the paired K/V expansion/reduction
experiment. It defaults false after Experiment 168. Both explicit operators stay available
for shape and Kernel diagnosis.

`--attention-gqa-value-broadcast true/false` controls the width-selective P×V+dP route.
It defaults false after Experiment 170. The operator APIs remain usable independently.

`--attention-gqa-forward-value-broadcast true/false` is the final forward-only variant.
It also defaults false after Experiment 171; the zero-stride model-routing family is closed.

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

For HIP Graph submission crossover measurements:

```bash
./build/hip-release/benchmarks/microllm_bench_hip_graph \
  --mode graph --nodes 128 --elements 4096 --warmup 5 --repetitions 20

python3 benchmarks/single_gpu/hip_graph_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_hip_graph \
  --output-directory /tmp/microllm-hip-graph
```

The benchmark owns every captured input/output allocation and reports setup separately from
replay. It is a runtime launch-overhead measurement, not evidence that the eager Transformer can
already be captured. See Experiment 173 for the allocation/Stream blockers.

To probe real caller-owned hipBLASLt shapes:

```bash
python3 benchmarks/single_gpu/hip_graph_gemm_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_hip_graph_gemm \
  --output-directory /tmp/microllm-hip-graph-gemm
```

This runner covers the Qwen/DeepSeek T512 projection shapes and keeps output addresses stable.
Experiment 174 retains the conformance/API but rejects repeated vendor-only Graph replay as an
end-to-end optimization.

Deferred-lifetime crossover:

```bash
python3 benchmarks/single_gpu/deferred_deallocation_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_deferred_deallocation \
  --output-directory /tmp/microllm-deferred-release
```

The safe control synchronizes before each temporary can be freed. The candidate synchronizes once
per explicit lifetime region and reports pending physical bytes. This is not comparable to an
unsafe no-sync chain.

Model-wide safe Stream matrix:

```bash
python3 benchmarks/single_gpu/scoped_deferred_model_matrix.py \
  --benchmark build/hip-release/benchmarks/microllm_bench_scoped_deferred_model \
  --output-directory /tmp/microllm-scoped-stream \
  --qwen-config /path/to/qwen/config.json \
  --qwen-weights /path/to/qwen/model.safetensors \
  --deepseek-config /path/to/deepseek/config.json \
  --deepseek-weights /path/to/deepseek/model.safetensors
```

The runner alternates fresh process order, compares every inference logit and the training loss/
updated parameter, and reports deferred physical bytes separately from logical engine peak.
Experiment 177 keeps the API for correctness but rejects default enablement because allocator
calls and retained bytes dominate every official row.

Stream-ordered allocator policy matrix:

```bash
python3 benchmarks/single_gpu/stream_ordered_allocator_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_stream_ordered_allocator \
  --output-directory /tmp/microllm-stream-ordered
```

The matrix distinguishes deferred lifetime, eager async pool calls, and captured Graph allocation
nodes. It reports address count, `3N+1` Graph-node structure and default-pool high/current bytes.
On the tested runtime, address reuse is real but neither async policy passes the speed gate.

Stable activation arena matrix:

```bash
python3 benchmarks/single_gpu/activation_arena_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_stream_ordered_allocator \
  --output-directory /tmp/microllm-activation-arena
```

This control allocates two aligned slots outside timing/capture. Arena Graph rows contain only
`N+1` compute Kernels and report setup plus calculated replay break-even. Replay wins are not
reported without the setup amortization count.

Official-shape FFN region:

```bash
python3 benchmarks/single_gpu/arena_ffn_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_arena_ffn \
  --output-directory /tmp/microllm-arena-ffn
```

This uses Qwen/DeepSeek hidden/intermediate dimensions, captures three hipBLASLt GEMMs plus
SwiGLU, compares every output element and reports shape-selective setup break-even.

Production BF16 FFN region:

```bash
python3 benchmarks/single_gpu/bf16_arena_ffn_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_bf16_arena_ffn \
  --output-directory /tmp/microllm-bf16-arena-ffn
```

This adds R1 decode shapes and distinguishes five-node direct FP32 output from six-node
caller-owned BF16 fallback. The runner reports measured engine allocation calls and retains
DeepSeek R32 Graph as a performance counterexample.

Complete-model eager Arena comparison:

```bash
python3 benchmarks/single_gpu/compare_bf16_ffn_arena_models.py \
  --manifest /absolute/path/to/hf-models.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory /tmp/microllm-bf16-ffn-arena-model
```

The CLI flag is `--bf16-ffn-arena true` and requires `--bf16-ffn true`. JSON reports
entry/hit/miss/capacity. Value tracing deliberately rejects the flag because traced layer details
use the diagnostic allocation-returning route rather than the timed workspace path.
Add `--arena-minimum-rows 512` to the runner, and
`--bf16-ffn-arena-minimum-rows 512` to the binary, to reproduce the selective gate. A bypass is
valid only when JSON reports zero entries/capacity/eligible calls and positive bypassed calls.
Use `--comparison-mode qkv` with the same runner to compare incremental QKV Arena on top of the
retained FFN threshold. Binary flags are `--bf16-qkv-arena` and
`--bf16-qkv-arena-minimum-rows`; Experiment 185 records why the model policy is rejected.

Allocation source×size attribution:

```bash
python3 benchmarks/single_gpu/hf_allocation_sources.py \
  --manifest /absolute/path/to/hf-models.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory /tmp/microllm-allocation-sources
```

The binary flag `--allocation-source-diagnostics true` requires one prefill and zero warm-up.
Records are logical engine requests, including allocator-cache reuse. Compare them with rocprofv3
malloc/free before drawing a backend-allocation conclusion.
Experiment 187 converts the selected source into `causal_gqa_attention_out_` and an opt-in
`--attention-core-arena` model path. Its failure is important: source bytes and removed logical
calls do not prove end-to-end speed once device Attention math dominates.

FP32 Attention solution screening:

```bash
python3 benchmarks/single_gpu/fp32_attention_solution_matrix.py \
  --binary build/hip-release/benchmarks/microllm_tune_fp32_attention_algorithms \
  --output-directory /tmp/microllm-fp32-attention-solutions
```

The tuner performs full-output correctness before Event/wall timing. Recommended indices are
version- and exact-layout-local; operator speedups are not model claims.

Complete-model FP32 solution gate:

```bash
python3 benchmarks/single_gpu/compare_fp32_attention_solutions.py \
  --manifest /absolute/path/to/hf-models.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory /tmp/microllm-fp32-attention-model
```

The runner separates QK-only, PV-only and both, rotates process order, compares complete logits,
and reports `fp32_solution_*` registry/cache/dispatch counters. The binary accepts
`--fp32-attention-qk-solution-index` and `--fp32-attention-pv-solution-index` only for explicit HIP
prefill experiments. No index is selected by default.

BF16 grouped QKV operator and model gates:

```bash
python3 benchmarks/single_gpu/bf16_grouped_qkv_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_bf16_grouped_qkv \
  --output-directory /tmp/microllm-bf16-grouped-qkv

python3 benchmarks/single_gpu/compare_bf16_grouped_qkv_models.py \
  --manifest /absolute/path/to/hf-models.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory /tmp/microllm-bf16-grouped-qkv-models
```

The first reports pointer-stable and per-call-reinitialized timing separately. The second requires
QKV Arena, checks complete logits/top tokens, plan hit/miss/dispatch counters, throughput and peak.
`--bf16-grouped-qkv-algorithm-index` is an explicit experiment flag; default inference registers
no grouped plan.
The expanded runner defaults to 64 candidates and final exact indices. JSON separates steady
throughput from `bf16_grouped_qkv_kernel_setup_ms` and argument setup. A warmed speedup is not a
TTFT claim when the first kernel setup exceeds the declared admission budget.
Use `--bf16-grouped-qkv-prewarm true` to move that setup before the measured request. Compare
`bf16_grouped_qkv_prewarm_ms` with `forward_ms`; their sum is the startup cost and must not be
reported as a request-only speedup.

## What remains

There is no Python `@profile` decorator yet. The current stable entry points are the
C++ RAII trace API, operator micro-benchmarks and the alignment runner. Future work must correlate ranges with
rocprof markers, support asynchronous HIP Event completion without per-range device
synchronization, and export Perfetto ranges directly.

Ad-hoc wall-clock timing around asynchronous kernels is not accepted evidence.
