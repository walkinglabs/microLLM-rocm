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

### Python, ROCTX, and GPU timeline

The Python profiling API can emit optional ROCTX ranges and fit its `perf_counter_ns`
clock to rocprof timestamps. Capture a real HIP add trace, then merge Python spans,
ROCTX ranges, and GPU Kernels into one Perfetto file:

```bash
result=/tmp/microllm-python-timeline
mkdir -p "$result"
HIP_VISIBLE_DEVICES=0 PYTHONPATH=python \
MICROLLM_LIBRARY="$PWD/build/hip-release/bindings/capi/libmicrollm.so" \
rocprofv3 --hip-trace --marker-trace --kernel-trace --output-format csv \
  --output-file python-unified --output-directory "$result" -- \
  python3 benchmarks/single_gpu/python_profile_timeline.py capture \
    --output "$result/profile.jsonl" --iterations 8 --overwrite

PYTHONPATH=python python3 benchmarks/single_gpu/python_profile_timeline.py merge \
  --profile "$result/profile.jsonl" \
  --marker "$result/python-unified_marker_api_trace.csv" \
  --kernel "$result/python-unified_kernel_trace.csv" \
  --hip-api "$result/python-unified_hip_api_trace.csv" \
  --calibration "$result/calibration.json" \
  --output "$result/unified.json"
```

The capture performs a separate ROCTX warm-up before the fitted ranges. Calibration
requires at least two ranges, scale within 1%, a call boundary no wider than 100us, and
maximum residual no larger than 50us. The curated three-process result and generated
quality chart are in
[`2026-08-26-python-roctx-gpu-perfetto`](results/2026-08-26-python-roctx-gpu-perfetto/).
This aligns clocks; it does not treat a Python wall span as asynchronous GPU completion.

### Asynchronous Python HIP Event completion

Use a start/finish HIP Event pair when a Python call only submits GPU work. The formal
runner keeps completion observation on a distinct thread and records HIP Event device
time separately from host observation time:

```bash
result=/tmp/microllm-event
mkdir -p "$result"
HIP_VISIBLE_DEVICES=0 PYTHONPATH=python \
MICROLLM_LIBRARY="$PWD/build/hip-release/bindings/capi/libmicrollm.so" \
rocprofv3 --hip-trace --marker-trace --kernel-trace --output-format csv \
  --output-file event --output-directory "$result" -- \
  python3 benchmarks/single_gpu/python_event_completion.py \
    --output "$result/profile.jsonl" --report "$result/report.json" --overwrite
```

The three-run result retains every HIP API row and proves that the formal route uses
Event record/query/synchronize with no device- or Stream-wide synchronize:
[`2026-08-26-python-hip-event-completion`](results/2026-08-26-python-hip-event-completion/).

### FP32 Attention request-batch invariance

The ordinary FP32 row-invariance tool changes GEMM `M`. Full-prefill Attention keeps
`M/N/K` fixed and changes the strided-batched descriptor instead. Use the dedicated
correctness-first harness for that case:

```bash
./build/hip-release/benchmarks/microllm_bench_fp32_attention_batch_invariance \
  --operation qk --sequence 2048 --heads 12 --kv-heads 2 --width 128 \
  --request-batches 1,2,4,8 --maximum-algorithms 64 \
  --workspace-bytes 33554432 --warmup 1 --repetitions 3
```

`qk` tests `M2048 N2048 K128`; `pv` tests `M2048 N128 K2048`. The backend batch
counts are 12/24/48/96. A B1 12-head input block is copied exactly to every request
row. Each common solution must pass a CPU sentinel, complete B1/default comparison,
and every output element of every repeated request block before its Event/wall timing
is recorded. `--inventory-only true` lists common indices without allocating the
large T2048 operands.

The reproducible two-operation driver is
[`fp32_attention_batch_invariance_matrix.py`](single_gpu/fp32_attention_batch_invariance_matrix.py).
An exact solution is not automatically admitted: the matrix separately requires its
minimum speedup across all requested batch sizes to remain at least 0.95.

After operator screening, the scoped model counterfactual is run by
[`fp32_prefill_attention_model_gate.py`](single_gpu/fp32_prefill_attention_model_gate.py).
Both policies keep the invariant Q/K/V projection solutions; only the candidate adds
QK/P×V indices. Candidate precision runs also export complete block-0 core values, while
paired performance runs alternate policy and batch order. Admission requires core
bitwise equality, robust complete-logit improvement, and at least 0.95× prefill speed in
every batch.

The final rebuttal uses
[`fp32_prefill_attention_selective_gate.py`](single_gpu/fp32_prefill_attention_selective_gate.py).
It keeps B1 on the default Attention descriptor and applies only the measured batch-local
QK/P×V winners at B2/B4/B8. It deliberately drops the bitwise-core claim and therefore
admits only on complete-logit Max/RMS plus end-to-end prefill and resource gates.

When rocWMMA 2.2 and OpenMP are available, the benchmark-only QK capability target
compares complete BF16×BF16→FP32 outputs against CPU, scalar HIP and hipBLASLt:

```bash
cmake --build build/hip-release --target microllm_bench_rocwmma_qk --parallel
./build/hip-release/benchmarks/microllm_bench_rocwmma_qk \
  --rows 512 --columns 512 --inner 128 \
  --tile 32 --waves-per-block 1 --warmup 10 --repetitions 50
```

The multi-process T16–2048/D64–128 runner is
`benchmarks/single_gpu/rocwmma_qk_matrix.py`. This is a capability gate for an online
Attention prototype; it does not register an operator or change model dispatch.

The admitted benchmark-only online causal-GQA prototype adds online max/sum and
rocWMMA PV without a global score Tensor:

```bash
./build/hip-release/benchmarks/microllm_bench_rocwmma_online_attention \
  --sequence 512 --heads 14 --kv-heads 2 --width 64 \
  --worker-threads 512 --warmup 5 --repetitions 20
```

`rocwmma_online_attention_matrix.py` runs Qwen/DeepSeek head grids from T32 to
T2048 and compares CPU, scalar fused, the candidate and the current framework
operator. The candidate remains outside public/model dispatch until fallback and
full-logit gates pass.

Screen and time the two matmul implementations in the required order:

```bash
./build/hip-release/benchmarks/microllm_tune_matmul \
  --m 128 --k 128 --n 128 --dtype fp32 \
  --warmup 3 --repetitions 10 --mode inference --accept false
```

Complete finite Max/RMS runs before timing. Only passing candidates receive HIP Event and
wall P50/P95. The default does not mutate the registry; explicit acceptance can write the
schema-versioned cache, but model-level regression remains a separate gate.

AdamW uses a separate in-place-state tuner. It clones the caller's parameter, gradient,
first/second moments and optional BF16 mirror, compares every updated element against
Scalar, and only then records default-Stream Event and wall P50/P95:

```bash
./build/hip-release/benchmarks/microllm_tune_adamw \
  --elements 802816 --mirror true --aligned true \
  --warmup 3 --repetitions 20 --mode training --accept false
```

Run the five-case fresh-process matrix with
`benchmarks/single_gpu/adamw_autotune_matrix.py`. Screening never changes `Auto`;
`--accept true --cache-output /path/cache.jsonl` is an explicit post-regression action.

Bias-gradient implementations have a complete-output micro benchmark:

```bash
./build/hip-release/benchmarks/microllm_bench_bias_gradient \
  --rows 512 --width 896 --implementation cooperative \
  --warmup 3 --repetitions 20
```

`benchmarks/single_gpu/bias_gradient_matrix.py` runs Scalar and cooperative paths in fresh
processes over the 16/32-row boundary and Qwen/DeepSeek widths. Auto keeps Scalar below
32 rows and selects the cooperative 2D reduction at or above the measured crossover.

Screen version-local hipBLASLt BF16 solution indices without installing a default:

```bash
./build/hip-release/benchmarks/microllm_tune_bf16_algorithms \
  --rows 512 --inner 896 --columns 896 --output-dtype fp32 \
  --maximum-algorithms 64 --workspace-bytes 33554432 \
  --warmup 2 --repetitions 5
```

`bf16_training_solution_matrix.py` repeats all eight Qwen/DeepSeek T512 forward shapes.
`hf_train_step --bf16-algorithms M:K:N:index,...` is an explicit process-local research
seam. It is not an environment-safe default or persistent cache.

For source/shape attribution without changing the default hot path, add
`--diagnostics-output /path/diagnostics.json` to `microllm_hf_train_step`. It records
Autograd first/add sources and Runtime strided-copy layouts only during measured steps.
`--tied-embedding-sparse-add true/false` provides same-binary A/B for the retained
Qwen tied-weight memory optimization.

Reprofile the current retained official-model training path with one command:

```bash
HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/profile_current_training.py \
  --manifest /path/to/hf-models.local.json \
  --binary build/hip-release/apps/microllm_hf_train_step \
  --output-directory /tmp/microllm-current-training-profile
```

The runner fixes B1T512, BF16 Linear, BF16 AdamW moments, the 1M hybrid threshold and
all accepted/rejected training controls. It collects two fresh rocprof processes per model
and derives one stable step from `(three-step - one-step) / 2`.

Reprofile the retained long-context cached-decode default with the corresponding
inference runner:

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/profile_current_cached_decode.py \
  --manifest /path/to/hf-models.local.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory /tmp/microllm-current-cached-profile \
  --model deepseek-r1-distill-qwen-1.5b \
  --context 2048 --batch 2 --decode-tokens 64 \
  --warmup 1 --many-step-count 3
```

The runner requires the measured scoped default to report `auto-enabled`, saves
Kernel/HIP API/copy/allocation CSV files, subtracts process load, and generates both
`profile-delta.json` and an autoresearch-style `profile-delta.svg`. The chart is a
visual index; raw CSV and the derived JSON remain the evidence used for decisions.
The curated result is
[`2026-08-25-post-materialized-deepseek-t2048-profile`](results/2026-08-25-post-materialized-deepseek-t2048-profile/).

Search the exact-order finalizer's physical-thread mapping without changing the
logical 256-lane reduction tree:

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/cached_attention_finalize_mapping_matrix.py \
  --benchmark build/hip-release/benchmarks/microllm_bench_cached_attention_stages \
  --output-directory /tmp/microllm-finalize-mapping \
  --models qwen2.5-0.5b,deepseek-r1-distill-qwen-1.5b \
  --sequences 512,2048 --batches 1,2 --cache-dtypes fp32,bf16 \
  --finalize-threads 64,128,256 --runs 2 --warmup 3 --repetitions 20
```

Every fresh process requires bitwise equality to the current fused context and
zero warm backend allocations. A mapped policy passes only when its median is at
least 1.05x in Event time and 1.02x in synchronized wall time versus 256 threads.
The runner emits raw JSON Lines, a case summary and `mapping.svg`; it never changes
the model's automatic policy.

Search P*V sequence parallelism while keeping score and softmax in exact current
order:

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/cached_attention_split_pv_matrix.py \
  --benchmark build/hip-release/benchmarks/microllm_bench_cached_attention_stages \
  --output-directory /tmp/microllm-split-pv \
  --models qwen2.5-0.5b,deepseek-r1-distill-qwen-1.5b \
  --sequences 512,2048 --batches 1,2 --cache-dtypes fp32,bf16 \
  --splits 1,2,4,8,16 --runs 2 --warmup 3 --repetitions 20
```

S1 must be bitwise equal to the exact-order materialized route and must remain in
the report as the extra-buffer/launch counterexample. Each larger split checks the
complete context before timing. A case reaches an official-model gate only at Event
1.05x and synchronized wall 1.02x with zero payload transfers and zero warm backend
allocations. The runner writes `raw.jsonl`, `summary.json`, and
`split-pv-search.svg`; it never changes Auto routing.

Test whether GQA query heads can reuse each value load while retaining every
head's exact position accumulation order:

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/cached_attention_gqa_value_reuse_matrix.py \
  --benchmark build/hip-release/benchmarks/microllm_bench_cached_attention_stages \
  --output-directory /tmp/microllm-gqa-value-reuse \
  --models qwen2.5-0.5b,deepseek-r1-distill-qwen-1.5b \
  --sequences 512,2048 --batches 1,2 --cache-dtypes fp32,bf16 \
  --tile-columns 8,16,32,64 --runs 2 --warmup 3 --repetitions 20
```

All tiles must be bitwise equal to materialized current before timing. The
candidate uses three logical outputs (scores, probabilities, context), reports
zero warm backend allocations, and passes only at Event 1.05x plus wall 1.02x.
The runner emits raw JSON Lines, a case summary, and `value-reuse.svg`; no model
or automatic route reads the tile selection.

Audit batch-shape numerical drift independently of performance timing:

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/audit_cached_cross_batch_logits.py \
  --manifest /path/to/hf-models.local.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory /tmp/microllm-cross-batch-logits
```

This saves selected decode steps 0/1/2, checks every batch row and host/device
argmax, and compares all vocabulary values against B1. To isolate the source at
step 0 across FP32 Linear, BF16 FFN, BF16 Attention, and combined BF16, run:

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/audit_cross_batch_precision.py \
  --manifest /path/to/hf-models.local.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory /tmp/microllm-cross-batch-precision
```

Both are diagnostic runs, not throughput rankings. They require two fresh
processes per case and report actual converted-tensor counts so a requested
precision island cannot silently fall back.

Screen the separate BF16 weight-gradient hypothesis before changing Autograd:

```bash
HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/bf16_weight_gradient_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_bf16_weight_gradient \
  --output-directory /tmp/microllm-bf16-weight-gradient
```

The candidate time includes FP32-input cast+transpose, FP32-gradient cast and the
BF16×BF16→FP32 GEMM. It reports complete-output finite/FP32 Max/RMS evidence and a
deterministic BF16 CPU sample gate. Small shapes are expected counterexamples; only
shape-selective winners may enter a later model A/B.

The rejected model route was tested with 20-step loss trajectories and complete gate/up
FP32-master comparison; its candidate-only runner has been removed. The reusable evidence
interfaces remain: `microllm_hf_train_step --loss-trajectory-output ...` writes measured
losses after timing, `--gate-up-parameters-output ...` writes selected FP32 safetensors,
and `microllm_compare_safetensors BASELINE CANDIDATE` compares every value. See Experiment 247.

Measure whether those cached logical allocations justify a workspace API:

```bash
HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/bf16_weight_gradient_workspace_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_bf16_weight_gradient \
  --output-directory /tmp/microllm-bf16-wgrad-workspace
```

The benchmark first primes the exact-size caching allocator, then separates Event and wall
time for the public allocating API and an equivalent caller-preallocated composition. It also
requires exactly three cache-reused allocations and zero backend allocations per public call.

For the two-GPU host-audit boundary:

```bash
ROCR_VISIBLE_DEVICES=0,1 python3 \
  benchmarks/distributed/data_parallel_verification_matrix.py \
  --binary build/rccl-release/apps/microllm_distributed_train \
  --output-directory /tmp/microllm-ddp-verification
```

It rotates every-step, final-step-only and disabled policies, requires exact loss trajectories,
and excludes the visible step-1 lazy setup from steady medians. Interval `1` remains the API/CLI
default regardless of the measured production speedup.

Sweep actual tiny-model bucket counts with final-step parameter verification:

```bash
ROCR_VISIBLE_DEVICES=0,1 python3 \
  benchmarks/distributed/data_parallel_bucket_matrix.py \
  --binary build/rccl-release/apps/microllm_distributed_train \
  --output-directory /tmp/microllm-ddp-buckets
```

This is a communication/bucket-count matrix, not overlap evidence. A one-bucket winner hands the
next milestone to a larger Model-S workload rather than manufacturing synthetic readiness events.

The distributed CLI now supports that workload directly:

```bash
ROCR_VISIBLE_DEVICES=0,1 \
  build/rccl-release/apps/microllm_distributed_train \
  --model model-s --context 32 --batch 1 --steps 3 \
  --bucket-bytes 4194304 --parameter-check-interval 3
```

Metrics include model/shape, exact parameter count, bucket parameter/element counts, stage times,
verification time and maximum per-rank engine peak bytes.

Run the natural multi-bucket matrix:

```bash
ROCR_VISIBLE_DEVICES=0,1 python3 \
  benchmarks/distributed/data_parallel_model_s_bucket_matrix.py \
  --binary build/rccl-release/apps/microllm_distributed_train \
  --output-directory /tmp/microllm-model-s-buckets
```

It scans 1/4/25 MiB, rotates process order, excludes step 1 from steady medians and requires
exact loss trajectories plus a final rank-parameter audit.

Distributed metrics also report exact bucket/average/unpacked Tensor counts, pack/unpack D2D
copies and communication-stage allocation/backend/cache-reuse deltas. These fields are the
admission gate for persistent reducer storage.

Gate in-place bucket averaging against its allocating control:

```bash
ROCR_VISIBLE_DEVICES=0,1 python3 \
  benchmarks/distributed/data_parallel_inplace_average_matrix.py \
  --binary build/rccl-release/apps/microllm_distributed_train \
  --output-directory /tmp/microllm-inplace-average
```

Both paths use the same 25 MiB/3-bucket Model-S workload and final-step parameter audit.

`microllm_bench_model` measures train or cache-backed generation throughput. Its
`tokens_per_second` excludes construction and warm-up; `tokens_per_second_with_setup`
includes construction, device transfer, optimizer allocation, and warm-up. Both are
reported so setup cannot disappear from the experiment.

Run the built-in single-device model ladder as one validated JSONL matrix:

```bash
python3 benchmarks/single_gpu/model_matrix.py \
  --benchmark build/hip-release/benchmarks/microllm_bench_model \
  --device hip \
  --profiles tiny,model-s,model-m \
  --modes train,generate \
  --output /tmp/microllm-single-gpu.jsonl
```

Each measurement includes exact parameter/FP32-weight bytes, model construction and
warm-up time, measured and setup-inclusive throughput, milliseconds per generated or
trained token, and current/peak/total engine-owned device bytes. The matrix checks
that metrics are finite and internally consistent; it deliberately does not use a
machine-noisy speed threshold as a correctness gate. See
[single-GPU model benchmarking](../docs/dev/single-gpu-benchmark.md).

Official checkpoints use a separate manifest because the files are intentionally not
stored in Git:

```bash
python3 benchmarks/single_gpu/hf_model_matrix.py \
  --manifest /path/to/hf-models.local.json \
  --infer-binary build/hip-release/apps/microllm_hf_infer \
  --train-binary build/hip-release/apps/microllm_hf_train_step \
  --device hip --modes infer,train \
  --output /tmp/microllm-hf-single-gpu.jsonl
```

Start from `benchmarks/single_gpu/hf_models.example.json`. Without
`--allow-unavailable`, any missing checkpoint/config/tokenizer input makes the command
nonzero. With that flag, missing inputs are emitted as `unavailable` and the matrix
summary is `incomplete`; they are never reported as passing measurements.

When an FP8 error trace identifies where drift grows but not where it originates,
screen every single-block FP32 counterfactual against the same complete-logit oracle:

```bash
HIP_VISIBLE_DEVICES=2 python3 \
  benchmarks/single_gpu/hf_fp8_layer_leave_one_out.py \
  --manifest /path/to/hf-models.local.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory /tmp/fp8-layer-search \
  --models qwen2.5-0.5b,deepseek-r1-distill-qwen-1.5b \
  --context 8 --physical-gpu-index 2
```

The runner fixes the retained E4/O-projection/dynamic-amax policy, executes one fresh
process per restored block, and ranks full-vocabulary Max/RMS. It is a screening tool:
single-process throughput is diagnostic, and the best layer must pass a separate
three-process short/long-context matrix before any policy is retained.

For phase-separated official inference across context, batch and cache modes:

```bash
ROCR_VISIBLE_DEVICES=1 python3 \
  benchmarks/single_gpu/hf_inference_shape_matrix.py \
  --manifest /path/to/hf-models.local.json \
  --micro-binary build/hip-release/apps/microllm_hf_infer \
  --pytorch-python /path/to/python-with-pytorch-rocm \
  --output-directory /tmp/microllm-inference-matrix \
  --suite standard \
  --decode-tokens 4 --warmup 1 --steps 2 --runs 3
```

For a service-focused cached-decode sweep with short/long contexts, batch scaling,
1/8/32/64-token outputs, KV capacity/utilization/waste, peak memory and paired PyTorch rows, use
`--suite serving --cases cached`. Warm-up executions are reported separately and excluded from
the measured throughput.

The runner separates prefill, cache preparation, steady cached decode and uncached
reference decode. It records engine/framework peak, resident weight policy, KV Storage
and active bytes, device-memory share, per-request memory, B1-relative batch scaling and
efficiency, tokens per peak GiB, exact greedy tokens and explicit `unsupported`/`oom` rows.
`smoke`, `standard`, `serving`, and `extended` suites cover increasingly wide short/long-context and
batch boundaries. See the [simple inference-matrix guide](../docs/dev/inference-matrix.zh-CN.md).

Prefill defaults to serving semantics: both frameworks project only the final hidden
position (`--prefill-logits-mode last`). Use `--prefill-logits-mode full` only when the
workload intentionally requires `[B,T,V]` logits. The selected mode is part of every raw
row and summary, so full-logits throughput cannot silently enter a TTFT comparison.

Run the independent Python/PyTorch ROCm baseline with the same built-in comparison
recipe, then compare machine-readable rows:

```bash
/path/to/python-with-pytorch-rocm \
  benchmarks/single_gpu/pytorch_model_matrix.py \
  --device cuda --profiles tiny,model-s,model-m --modes train,generate \
  --measurement-profile comparison \
  --output /tmp/pytorch-builtins.jsonl

python3 benchmarks/single_gpu/compare_frameworks.py \
  --kind builtins \
  --microllm /tmp/microllm-builtins.jsonl \
  --pytorch /tmp/pytorch-builtins.jsonl \
  --output /tmp/framework-comparison.jsonl
```

The PyTorch runner uses an idiomatic eager Decoder with PyTorch SDPA, RMSNorm, RoPE,
MHA/GQA, SwiGLU, AdamW, and a real K/V cache. Each row runs in a fresh Python process.
See [PyTorch performance comparison](../docs/dev/pytorch-benchmark.md) for fairness,
memory-counter, warm-up, and AMDSMI fallback rules.

Probe BF16 GroupedGemm with the exact Qwen/DeepSeek T512 Q/K/V widths, including direct-FP32
capability rejection, BF16-output casts, pointer-stable timing and per-call reinitialization:

```bash
python3 benchmarks/single_gpu/bf16_grouped_qkv_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_bf16_grouped_qkv \
  --output-directory /tmp/microllm-bf16-grouped-qkv
```

Use `compare_bf16_grouped_qkv_models.py` for the complete-logit, throughput and peak-memory
gate. Grouped plan indices are exact-environment experiments and are never selected by default.

Probe FP32 QKV/gate-up weight gradients before changing Autograd:

```bash
python3 benchmarks/single_gpu/grouped_weight_gradient_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_grouped_weight_gradient \
  --output-directory /tmp/microllm-grouped-weight-gradient
```

It covers direct `N,T` and one-shared-transpose `N,N` layouts. A zero-supported row is a valid
capability result, not a fallback performance measurement.

Gate the explicit Autograd external leaf pool across complete Tiny/Model-S gradients,
rotated process order, allocation counters, measured peak and Event/wall timing:

```bash
python3 benchmarks/single_gpu/external_gradient_pool_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_external_gradient_pool \
  --output /tmp/microllm-external-gradient-pool \
  --runs 3 --warmup 1 --repetitions 5
```

The timed scope is `zero_grad + forward + backward`; full-gradient host verification follows
timing. Ratios are `baseline / external`. Current MI300X evidence keeps the API explicit for
interop and rejects it as the default training policy.

If grouped FP32 capability is absent, the packed counterfactual includes every D2D pack operation:

```bash
python3 benchmarks/single_gpu/packed_weight_gradient_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_packed_weight_gradient \
  --output-directory /tmp/microllm-packed-weight-gradient
```

Screen exact rank-2 gate/up weight-gradient solution indices before a model gate:

```bash
python3 benchmarks/single_gpu/fp32_weight_gradient_solution_matrix.py \
  --binary build/hip-release/benchmarks/microllm_tune_fp32_weight_gradient_algorithms \
  --output-directory /tmp/microllm-fp32-wgrad-solutions
```

The training CLI flag `--fp32-gate-up-weight-gradient-solution-index` is diagnostic-only and
reports exact registry hits/dispatches. Published evidence rejects both current model indices.

After the grouped and BTHD policies pass independently, gate direct BF16 Q/K consumption
through the fused bias+RoPE boundary with fresh-process medians:

```bash
python3 benchmarks/single_gpu/compare_inference_bthd_bf16_qk.py \
  --manifest /path/to/hf-models.local.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory /tmp/microllm-bthd-bf16-qk
```

The default is five processes per policy/model because the measured gain is about 2%.
The runner requires complete-logit equality, exact retained-dispatch counts, unchanged
peak, and a 1.01x per-model speed gate. This is an explicit exact-environment experiment,
not a portable default.

Use `compare_inference_bthd_bf16_qk_shapes.py` for the B1/T256,
B1/T1024 and B2/T512 expansion. Its default five-process matrix also checks
every batch row's complete logits and top token.

Use `compare_causal_softmax_threads.py` with
`microllm_bench_causal_softmax` to compare explicit 256/128-thread row
implementations. The runner gates complete output before Event timing and reports
universal and T512 performance separately; it does not route models automatically.

Use `compare_bf16_repeat.py` with `microllm_bench_bf16_repeat` to compare
device-native BF16 cast-plus-repeat against the fused typed primitive on official
V shapes. The timed region rejects any H2D/D2H payload transfer.

Capture HIP API, kernel, memory, and RCCL-ready runtime traces with the locally
installed rocprofv3 interface:

```bash
./scripts/profile_hip.sh /tmp/microllm-trace -- \
  ./build/benchmarks/microllm_bench_model \
  --mode train --model tiny --device hip --steps 5 --warmup 1 \
  --batch 1 --context 8 --new-tokens 8
```

Measure the optional PyTorch ROCm dispatcher in fresh processes with rotated order:

```bash
/path/to/rocm-venv/bin/python \
  benchmarks/single_gpu/pytorch_custom_op_rocm_matrix.py \
  --library build/torch-rocm/bindings/torch/libmicrollm_torch_ops.so \
  --output /tmp/microllm-pytorch-rocm-custom-ops \
  --runs 3 --warmup 5 --repetitions 25
```

The matrix covers add/multiply across FP32/FP16/BF16 at 4K/1M/16M elements and two
Autograd branches. Speed is `Torch / microLLM`; complete Max/RMS/loss and allocator peak
are mandatory even when the result is a performance rejection.

To reproduce the scalar/broad/selective vector16 admission decision, run the same matrix
against each build and compare their raw worker records:

```bash
python3 benchmarks/single_gpu/compare_pytorch_custom_op_vector16.py \
  --baseline benchmarks/results/2026-08-26-pytorch-rocm-custom-ops \
  --broad benchmarks/results/2026-08-26-pytorch-rocm-custom-ops-vector16 \
  --selective benchmarks/results/2026-08-26-pytorch-rocm-custom-ops-vector16-selective \
  --output /tmp/microllm-vector16-comparison.json
```

The gate requires all values/gradients exact, equal peaks, every FP16/BF16 16M row at least
1.05× scalar, and FP32 bandwidth non-regression. The broad result is intentionally retained.
