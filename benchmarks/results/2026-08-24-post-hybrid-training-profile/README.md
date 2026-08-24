# Post-hybrid training profile

Experiment 216 profiles the retained 1M Hybrid BF16 AdamW path. For each official model it
subtracts `load + 1 step` Kernel statistics from `load + 3 steps`, then divides by two. This removes
weight loading, mirror preparation, library setup and first-allocation work.

## Result

| Model | Kernel/step | GEMM | AdamW | GEMM + AdamW | Calls/step |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 32.117 ms | 59.33% | 12.82% | 72.15% | 1,698 |
| DeepSeek Distill 1.5B | 72.906 ms | 63.81% | 17.61% | 81.41% | 2,037 |

Compared with Experiment 213, AdamW Kernel time improves `1.372×/1.293×` and total per-step Kernel
time improves `1.044×/1.058×`. Qwen now launches 73 large per-tensor updates plus one small-tensor
update; DeepSeek launches 142 plus one. The next major target is therefore training GEMM grouping or
exact shape planning, not another AdamW threshold.

`qwen/` and `deepseek/` contain the compact one/three-step Kernel stats and deterministic
`profile-delta.json`. The complete per-dispatch traces remain external because the compact files are
sufficient to reproduce every reported total and category.

## Reproduce

Collect each one/three-step pair with `rocprofv3 --kernel-trace --stats --output-format csv`, then:

```bash
python3 benchmarks/single_gpu/profile_step_delta.py \
  --one-step /path/to/one_kernel_stats.csv \
  --many-step /path/to/three_kernel_stats.csv \
  --many-step-count 3 \
  --output-directory /tmp/training-profile-delta
```

The parser rejects negative call deltas, classifies softmax/repeat separately, and has a registered
contract test.
