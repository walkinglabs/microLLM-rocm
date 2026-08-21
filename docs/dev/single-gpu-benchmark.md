# Single-GPU model memory and performance matrix

## What this gate proves

`Benchmark.HipModelMatrix` executes both training and cache-backed generation for
three repository-owned model sizes on one visible AMD GPU:

| Profile | Parameters | FP32 weights | Role |
|---|---:|---:|---|
| `tiny` | 5,712 | 22,848 bytes | fast graph/measurement smoke |
| `model-s` | 15,586,176 | 62,344,704 bytes | approximately 64 MB tier |
| `model-m` | 31,334,912 | 125,339,648 bytes | approximately 128 MB tier |

For every profile and mode, the runner requires:

- the executable parameter count and FP32 weight bytes to equal the profile contract;
- finite loss/output guard and positive measured token count;
- finite positive `tokens_per_second` and `milliseconds_per_token`;
- peak engine device bytes not below current bytes or the model's FP32 weights;
- device name, architecture, HIP runtime and driver metadata;
- construction, warm-up, measured, and setup-inclusive wall time;
- current, peak, and cumulative engine-owned CPU/device allocation bytes.

## Run it

```bash
cmake --preset hip-release
cmake --build --preset hip-release --parallel
ctest --test-dir build/hip-release \
  -R '^Benchmark.HipModelMatrix$' \
  --output-on-failure -V
```

The raw build-tree artifact is
`build/hip-release/benchmarks/hip-model-matrix.jsonl`. A curated run may be copied
under `benchmarks/results/` only with its exact configuration and limitations.

## Why performance is recorded but not thresholded

A correctness test should fail when a parameter count is wrong, memory accounting is
impossible, a loss is non-finite, or no work was measured. It should not fail merely
because a shared GPU was temporarily slower. Performance regression thresholds require
dedicated, controlled runners and repeated baselines.

The CTest gate therefore records performance on every run but treats it as descriptive
evidence. Publication claims still require repeated measurements and profiler traces.

## What peak memory means

`device_peak_engine_bytes` comes from the microLLM allocation tracker. It includes
Tensor and engine workspace allocations performed through the runtime allocator. It
does not automatically include memory privately allocated by the HIP runtime,
hipBLASLt, a driver cache, another process, or an external framework.

For a release memory claim, pair this number with rocprof memory tracing or an external
device-memory sampler. Do not rename engine-owned peak bytes to total board peak memory.

## Common external models

The repository also has measured official-weight paths for:

- Qwen2.5-0.5B, 494,032,768 parameters;
- DeepSeek-R1-Distill-Qwen-1.5B, 1,777,088,000 parameters.

Those checkpoints are not stored in Git and therefore cannot be mandatory for every
CTest runner. `hf_model_matrix.py` consumes an explicit manifest and records inference
and/or a full backward/AdamW step. Start from:

```text
benchmarks/single_gpu/hf_models.example.json
```

Every successful inference row requires strict Tensor count, exact greedy tokens,
positive prefill/decode timing, and current/peak/total engine memory. Every successful
training row additionally requires a finite loss, a changed parameter, positive
tokens/s, and zero optimizer Tensor-payload H2D/D2H calls.

If any config, weight, vocab, or merges path is missing, the row is `unavailable` and
the summary is `incomplete`. The regular CPU CTest exercises exactly this negative
case. `--allow-unavailable` only lets that explicit incomplete report exit zero; it
does not relabel the rows.

Run real local files without that flag:

```bash
python3 benchmarks/single_gpu/hf_model_matrix.py \
  --manifest /path/to/hf-models.local.json \
  --infer-binary build/hip-release/apps/microllm_hf_infer \
  --train-binary build/hip-release/apps/microllm_hf_train_step \
  --device hip --modes infer,train \
  --output /tmp/microllm-hf-matrix.jsonl
```

For an optimizer implementation experiment, `microllm_hf_train_step` also accepts
`--adamw-implementation auto|scalar|vectorized`. Published model results use `auto`;
explicit modes are counterfactual candidates and must be labeled as such.

The first curated official-weight result is under
`benchmarks/results/2026-08-20-mi300x-single-gpu-hf-matrix/`.

Matched Python/PyTorch ROCm runs and automatic ratios are documented in
[PyTorch performance comparison](pytorch-benchmark.md). A microLLM-only row is not
considered a framework comparison.
