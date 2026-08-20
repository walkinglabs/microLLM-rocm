# Experiment 036 raw evidence

`raw.jsonl` contains one FP32 full-logit reference and three plan-cache candidate processes
per model. `summary.json` compares medians with Experiment 034 and the fixed Experiment 031
PyTorch BF16 reference.

```bash
python3 benchmarks/single_gpu/run_bf16_attention_models.py \
  --manifest /path/to/local-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --baseline-summary docs/optimization-log/experiments/034-data/summary.json \
  --baseline-kind attention \
  --pytorch-summary docs/optimization-log/experiments/031-data/summary.json \
  --raw-output docs/optimization-log/experiments/036-data/raw.jsonl \
  --summary-output docs/optimization-log/experiments/036-data/summary.json \
  --runs 3
```
