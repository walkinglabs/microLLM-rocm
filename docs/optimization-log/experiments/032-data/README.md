# Experiment 032 raw evidence

`raw.jsonl` contains twelve microLLM rows:

```text
2 models × 2 policies × 3 independent processes
```

The PyTorch full-BF16 reference is unchanged and is reused from Experiment 031. To rerun:

```bash
python3 benchmarks/single_gpu/run_bf16_ffn_models.py \
  --manifest /path/to/local-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --raw-output docs/optimization-log/experiments/032-data/raw.jsonl \
  --summary-output docs/optimization-log/experiments/032-data/summary.json \
  --runs 3 --prefill-warmup 2 --prefill-steps 5
```

The local manifest keeps the revisions/token IDs from
`benchmarks/single_gpu/hf_models.example.json` and changes only checkpoint paths.
