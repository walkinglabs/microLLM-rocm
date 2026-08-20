# Experiment 042 raw evidence

`raw.jsonl` contains 24 measurements:

```text
Qwen2.5-0.5B
× shapes 1x3, 2x3, 1x32, 1x128
× microLLM / PyTorch
× 3 fresh processes
```

Odd runs execute microLLM first; even runs execute PyTorch first. This reduces, but cannot
eliminate, shared-GPU time drift. Reproduce with:

```bash
python3 benchmarks/single_gpu/hf_training_shape_matrix.py \
  --manifest /path/to/local-manifest.json \
  --micro-binary build/hip-release/apps/microllm_hf_train_step \
  --pytorch-python /path/to/rocm-python \
  --pytorch-runner benchmarks/single_gpu/pytorch_hf_model_matrix.py \
  --output-directory docs/optimization-log/experiments/042-data \
  --models qwen2.5-0.5b \
  --shapes 1x3,2x3,1x32,1x128 \
  --precision bf16 --warmup 1 --steps 2 --runs 3 \
  --allow-amdsmi-fallback
```

`summary.json` is generated from per-framework medians. Engine allocator peak and
`torch.cuda.max_memory_allocated` have different allocator scopes; their ratio is useful
but is not a byte-for-byte ownership proof.
