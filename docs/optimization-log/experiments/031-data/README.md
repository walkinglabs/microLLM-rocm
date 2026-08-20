# Experiment 031 raw evidence

`raw.jsonl` has 18 rows:

```text
2 models × 3 process runs ×
  (microLLM FP32 + microLLM single-representation BF16 FFN + PyTorch full BF16)
```

`summary.json` contains medians and ratios. `preparation-smoke.jsonl` separately records
the one-time transactional conversion peak, because serving measurements reset peak memory
after warm-up.

## Reproduce

Create a local manifest from `benchmarks/single_gpu/hf_models.example.json`. Keep the
pinned revisions, token IDs and expected generated IDs; replace only local file paths.

```bash
python3 benchmarks/single_gpu/run_bf16_ffn_models.py \
  --manifest /path/to/local-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --raw-output docs/optimization-log/experiments/031-data/raw.jsonl \
  --summary-output docs/optimization-log/experiments/031-data/summary.json \
  --runs 3 --prefill-warmup 2 --prefill-steps 5 \
  --pytorch-python /path/to/rocm-python \
  --pytorch-runner benchmarks/single_gpu/pytorch_hf_model_matrix.py
```

Local checkpoint paths are not committed. Model identity and expected outputs are.
