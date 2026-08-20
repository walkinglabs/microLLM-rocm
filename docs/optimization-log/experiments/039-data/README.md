# Experiment 039 raw evidence

Six candidate rows cover two models and three independent processes. The retained FP32,
BF16 and PyTorch baselines come from Experiment 037.

```bash
python3 benchmarks/single_gpu/run_bf16_training_candidate.py \
  --manifest /path/to/local-manifest.json \
  --binary build/hip-release/apps/microllm_hf_train_step \
  --baseline-summary docs/optimization-log/experiments/037-data/summary.json \
  --raw-output docs/optimization-log/experiments/039-data/raw.jsonl \
  --summary-output docs/optimization-log/experiments/039-data/summary.json \
  --runs 3
```

The candidate source was removed after the keep gate failed.
