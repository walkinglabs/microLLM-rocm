# Current retained training profile

Experiment 244 re-runs the retained B1T512 BF16 training path from the current
binary. Each official model uses two fresh rocprof processes:

```text
load + 1 training step
load + 3 training steps
-----------------------
difference / 2 = one stable training step
```

## Result

| Model | Kernel/step | GEMM | AdamW | Calls/step | Versus Experiment 216 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 31.327 ms | 58.56% | 13.22% | 1,698 | 1.0252× |
| DeepSeek Distill 1.5B | 71.873 ms | 63.43% | 18.16% | 2,037 | 1.0144× |

All four application records update a parameter, keep optimizer D2H at zero and
emit the exact retained descriptor metadata bytes. There are no negative Kernel-call
deltas. The hotspot ordering is unchanged: training GEMM remains the next architecture
target; AdamW threshold tuning stays closed.

The Kernel delta is a device-time attribution tool, not an end-to-end throughput claim.
The application JSON records retain measured wall time, loss, memory and transfer fields.

## Reproduce

```bash
HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/profile_current_training.py \
  --manifest /path/to/hf-models.local.json \
  --binary build/hip-release/apps/microllm_hf_train_step \
  --output-directory /tmp/microllm-current-training-profile
```

