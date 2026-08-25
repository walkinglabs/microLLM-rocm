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
