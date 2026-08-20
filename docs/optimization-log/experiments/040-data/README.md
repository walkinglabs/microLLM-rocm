# Experiment 040 raw evidence

`raw.jsonl` contains six candidate records: two pinned Hugging Face models times three
independent processes. `summary.json` contains their medians and ratios against Experiment
037. The model revision fields identify the pinned model artifacts; the source change is
commit `f053816`.

Reproduce with a local manifest whose paths point to the pinned config and safetensors:

```bash
python3 benchmarks/single_gpu/run_bf16_training_candidate.py \
  --candidate-name weight_mirrors \
  --manifest /path/to/local-manifest.json \
  --binary build/hip-release/apps/microllm_hf_train_step \
  --baseline-summary docs/optimization-log/experiments/037-data/summary.json \
  --raw-output docs/optimization-log/experiments/040-data/raw.jsonl \
  --summary-output docs/optimization-log/experiments/040-data/summary.json \
  --runs 3
```

The comparison protocol is two unmeasured warm-up steps followed by five measured training
steps. Peak bytes use the engine allocator counter. Mirror bytes are reported separately
and are not hidden inside the FP32 weight number.
