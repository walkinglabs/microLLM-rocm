# Experiment 061 evidence

- `route-unused-pilot.jsonl`: operator routing alone did not reach the model path.
- `integrated-pilot.jsonl`: first Qwen T512 model integration result.
- `fallback128.jsonl`: short-sequence Qwen process.
- `formal/`: Qwen/DeepSeek × T512/T1024 × two frameworks × three fresh processes;
  prefill-only case selection, one warm-up and two measured iterations.
- `profile-before/`: Qwen T512 model still using two readable Attention matmuls/layer.
- `profile-after/`: model reuses public causal GQA and batched hipBLASLt.
- `comparison.json` and `profile-summary.json`: machine-checkable keep contracts.

The formal baseline is Experiment 060 core/long-warm on physical visibility 1/3. The
candidate formal matrix runs on visibility 1 and reruns PyTorch in the same paired window.
