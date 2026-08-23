# Post-bias training phase profile

Experiment 159 separates load-only and per-training-step kernels after Experiment 158.

- `one-step-kernel-stats.csv`: load plus one training step;
- `three-step-kernel-stats.csv`: load plus one warm-up and two measured steps;
- `profile-delta.json`: `(three-step - one-step) / 2` calls and duration;
- `verification.json`: machine-readable next-hotspot decision.

The subtraction removes kernels whose calls do not grow with training steps. In
particular, BF16-to-FP32 cast-transpose is a checkpoint-load cost, not a measured
training target. The derived per-step profile assigns 53.47% of Kernel time to
hipBLASLt GEMMs. AdamW is second, but its available candidates were closed by
Experiment 157; the next open implementation node is exact training-GEMM solution
enumeration and persistence.

Recompute:

```bash
python3 benchmarks/single_gpu/profile_step_delta.py \
  --one-step /path/one-step-kernel-stats.csv \
  --many-step /path/three-step-kernel-stats.csv --many-step-count 3 \
  --output-directory /tmp/profile-delta
```
