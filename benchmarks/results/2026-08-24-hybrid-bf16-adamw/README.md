# Hybrid BF16 AdamW evidence

Experiment 215 combines only BF16-moment tensors at or below a size threshold. Larger tensors keep
the independent vectorized Kernel that won the bandwidth experiment.

## Threshold sweep

Every threshold directory contains three fresh processes per model and policy. The formal directory
contains five. All runs use BF16 Linear + FP32 master, batch 1, context 512, one warm-up and two
measured steps.

| Threshold | Qwen selected tensors/elements | DeepSeek selected tensors/elements | Result |
|---:|---:|---:|---|
| 4,096 | 121 / 71,552 | 141 / 144,896 | both gates pass |
| 65,536 | 121 / 71,552 | 141 / 144,896 | identical selected set; timing difference is noise |
| 262,144 | 169 / 5,576,576 | 141 / 144,896 | both gates pass |
| 1,048,576 | 217 / 44,111,744 | 197 / 22,164,992 | selected local optimum |
| 4,194,304 | 217 / 44,111,744 | 253 / 154,285,568 | lower two-model geometric gain |
| 16,777,216 | 289 / 357,898,112 | 337 / 1,310,340,608 | DeepSeek optimizer 0.896× and E2E 0.980×; reject |

The final five-process 1 Mi-element threshold reaches Qwen/DeepSeek optimizer
`1.2404×/1.2631×` and end-to-end `1.0490×/1.0528×` versus FP32 moments. Relative to the retained
per-tensor BF16-moment route from Experiment 214, throughput improves another
`1.0245×/1.0185×`.

`verification.json` records the complete CPU, sanitizer, PyTorch, HIP, RCCL, coverage and
test-registration release gates for this exact source state.

## Reproduce

```bash
HIP_VISIBLE_DEVICES=0 python3 benchmarks/single_gpu/adamw_moment_matrix.py \
  --binary build/hip-release/apps/microllm_hf_train_step \
  --qwen-config /path/to/qwen/config.json \
  --qwen-weights /path/to/qwen/model.safetensors \
  --deepseek-config /path/to/deepseek/config.json \
  --deepseek-weights /path/to/deepseek/model.safetensors \
  --output-directory /tmp/hybrid-bf16-adamw \
  --runs 5 --warmup 1 --steps 2 --batch 1 --context 512 \
  --bf16-multi-tensor-threshold 1048576
```

The public CLI uses `auto` by default after BF16 moments are selected. `auto` currently resolves to
1,048,576 on HIP; `0` restores the per-tensor counterfactual and a positive integer is an explicit
research override.
