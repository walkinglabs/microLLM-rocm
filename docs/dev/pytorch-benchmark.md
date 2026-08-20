# Python/PyTorch ROCm performance comparison

## What is compared

The baseline is implemented in Python and runs PyTorch directly; the microLLM core does
not depend on LibTorch. Two matrices exist:

1. repository-owned tiny, Model-S and Model-M architecture profiles;
2. official Qwen2.5-0.5B and DeepSeek-R1-Distill-Qwen-1.5B checkpoints through
   Transformers.

The built-in PyTorch Decoder uses eager FP32 `nn.Embedding`, Linear, RMSNorm, RoPE,
MHA/GQA, `scaled_dot_product_attention`, SwiGLU, AdamW and K/V cache. It has the same
parameter counts and workload fields as microLLM. It intentionally uses idiomatic
PyTorch kernels rather than reimplementing microLLM's readable HIP kernels in Python.

Official HF rows use exactly the same checkpoint, token IDs, predicted-token count,
greedy expected tokens and FP32 compute. Qwen tied embeddings expose 291 PyTorch
state-dict names for a 290-Tensor checkpoint; both counts are recorded separately.

## Comparison-grade versus smoke

The CI `smoke` profile favors a short runtime. It is not used for framework ratios.
The `comparison` profile is:

| Model | Mode | Warm-up | Repetitions | Context |
|---|---|---:|---:|---:|
| tiny | train/generate | 3 | 10 | 8 |
| Model-S | train/generate | 1 | 3 | 2 / 4 |
| Model-M | train/generate | 1 | 3 | 1 / 4 |

Every model/mode runs in a fresh process on both sides, preventing a previous row from
pre-initializing allocator or kernel state. The comparator refuses mismatched parameter
count, dtype, batch, context, steps, warm-up, token count, or measurement profile.

The official HF matrix remains a one-run inference and one-step training smoke. Its
training ratio includes first-step kernel setup and is not a steady-state training
claim.

## Run the built-in comparison

```bash
python3 benchmarks/single_gpu/model_matrix.py \
  --benchmark build/hip-release/benchmarks/microllm_bench_model \
  --device hip --profiles tiny,model-s,model-m --modes train,generate \
  --measurement-profile comparison --output /tmp/microllm.jsonl

/path/to/python-with-pytorch-rocm \
  benchmarks/single_gpu/pytorch_model_matrix.py \
  --device cuda --profiles tiny,model-s,model-m --modes train,generate \
  --measurement-profile comparison --output /tmp/pytorch.jsonl

python3 benchmarks/single_gpu/compare_frameworks.py \
  --kind builtins --microllm /tmp/microllm.jsonl \
  --pytorch /tmp/pytorch.jsonl --output /tmp/comparison.jsonl
```

## Run official HF models

Use the same local manifest for both runners:

```bash
/path/to/python-with-pytorch-rocm \
  benchmarks/single_gpu/pytorch_hf_model_matrix.py \
  --manifest /path/to/hf-models.local.json \
  --device cuda --modes infer,train --output /tmp/pytorch-hf.jsonl

python3 benchmarks/single_gpu/compare_frameworks.py \
  --kind huggingface --microllm /tmp/microllm-hf.jsonl \
  --pytorch /tmp/pytorch-hf.jsonl --output /tmp/hf-comparison.jsonl
```

Missing checkpoints remain `unavailable`, never `pass`.

## Memory counters are not identical

The ratio is explicitly named:

```text
microLLM device_peak_engine_bytes
÷ PyTorch torch.cuda.max_memory_allocated
```

Both are allocator-level measurements, but they do not have identical internals.
PyTorch also reports peak reserved bytes. Neither counter automatically includes every
driver allocation or another process. The comparison JSON retains the counter scope so
the ratio cannot be mistaken for total board-memory usage.

## Current ROCm environment workaround

The tested PyTorch wheel is `2.11.0+rocm7.13.0rc2`. Its HIP runtime API sees four
MI300X devices, while the wheel's AMDSMI discovery returns zero. The runner refuses to
hide that mismatch. On this machine it was invoked with `--allow-amdsmi-fallback`, which
records:

```text
amdsmi_zero_fallback_to_hip_runtime
```

This uses PyTorch's underlying HIP runtime count and is not needed on a healthy AMDSMI
installation.
