# Experiment 034 raw evidence

- `raw.jsonl`: one FP32 full-logit reference plus three shared-cast candidate processes per
  model (8 records total);
- `summary.json`: candidate medians versus Experiment 032 BF16-FFN and Experiment 031
  PyTorch full-BF16 references;
- `naive-pilot.jsonl`: one official-model row per model before Q/K/V shared input cast.

```bash
python3 benchmarks/single_gpu/run_bf16_attention_models.py \
  --manifest /path/to/local-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --baseline-summary docs/optimization-log/experiments/032-data/summary.json \
  --pytorch-summary docs/optimization-log/experiments/031-data/summary.json \
  --raw-output docs/optimization-log/experiments/034-data/raw.jsonl \
  --summary-output docs/optimization-log/experiments/034-data/summary.json \
  --runs 3
```

Checkpoint paths are local; revisions, token IDs and expected tokens remain pinned in the
repository example manifest.
