# BF16 AdamW moment evidence

This directory preserves every stage of Experiment 214, including failed candidates.

## Layout

- root `training.jsonl` / `summary.json`: first five-process per-tensor run before the
  asynchronous optimizer timing boundary was corrected;
- `pilot-multi/`: one-element-per-thread BF16 multi-tensor candidate;
- `pilot-vector-multi/`: four-elements-per-thread BF16 multi-tensor candidate with corrected
  optimizer timing; rejected because it regressed the official model gate;
- `pilot-per-tensor-timed/`: corrected three-process per-tensor pilot;
- `formal/`: retained five-process per-tensor comparison and authoritative summary.
- `verification.json`: complete CPU/sanitizer/PyTorch/HIP/RCCL and clean coverage gates.

The authoritative status is `partial_keep`: required accuracy, end-to-end throughput, exact
state-memory and peak-memory gates pass for both models. Qwen reaches only `1.0687×` on the
separate `1.10×` optimizer stretch gate, while DeepSeek reaches `1.1964×`.

## Reproduce

```bash
HIP_VISIBLE_DEVICES=0 python3 benchmarks/single_gpu/adamw_moment_matrix.py \
  --binary build/hip-release/apps/microllm_hf_train_step \
  --qwen-config /path/to/qwen/config.json \
  --qwen-weights /path/to/qwen/model.safetensors \
  --deepseek-config /path/to/deepseek/config.json \
  --deepseek-weights /path/to/deepseek/model.safetensors \
  --output-directory /tmp/bf16-adamw-moments \
  --runs 5 --warmup 1 --steps 2 --batch 1 --context 512
```

The checkpoints are intentionally not committed. The JSON records retain device/runtime identity,
workload shape, warm-up, measured steps, complete loss endpoints, state bytes, transfer counters,
optimizer timing boundary, throughput and engine peak memory.
