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

Capture HIP API, kernel, memory, and RCCL-ready runtime traces with the locally
installed rocprofv3 interface:

```bash
./scripts/profile_hip.sh /tmp/microllm-trace -- \
  ./build/benchmarks/microllm_bench_model \
  --mode train --model tiny --device hip --steps 5 --warmup 1 \
  --batch 1 --context 8 --new-tokens 8
```
