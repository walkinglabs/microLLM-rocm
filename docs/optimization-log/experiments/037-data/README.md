# Experiment 037 raw evidence

`raw.jsonl` contains:

```text
2 models × 3 policies × 3 independent processes = 18 rows
```

Policies are microLLM FP32, microLLM BF16 Linear with FP32 masters, and PyTorch BF16
autocast with FP32 parameters. Every process uses two warm-ups and five measured updates.

```bash
python3 benchmarks/single_gpu/run_bf16_training_models.py \
  --manifest /path/to/local-manifest.json \
  --micro-binary build/hip-release/apps/microllm_hf_train_step \
  --pytorch-python /path/to/rocm-python \
  --pytorch-runner benchmarks/single_gpu/pytorch_hf_model_matrix.py \
  --raw-output docs/optimization-log/experiments/037-data/raw.jsonl \
  --summary-output docs/optimization-log/experiments/037-data/summary.json \
  --runs 3
```

`pytorch-native-bf16-failure.json` preserves why native BF16 parameters are not the matched
master-weight reference at this learning rate.
