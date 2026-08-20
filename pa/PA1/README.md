# PA1 — 一个可复现的性能失败

Choose one operator or end-to-end path. Do not start by editing a kernel.

## Required workflow

```text
fixed GPU/ROCm/model/shape/dtype
→ CPU/HIP correctness gate
→ warm-up and repeated baseline JSONL
→ rocprofv3 trace
→ one primary hypothesis
→ one minimal change
→ numerical regression
→ repeated operator and end-to-end measurements
→ retained counterexample
```

Useful commands:

```bash
MICROLLM_BENCH_DEVICE=hip "$MICROLLM_ENGINE_DIR/scripts/run_benchmarks.sh"
"$MICROLLM_ENGINE_DIR/scripts/profile_hip.sh" /tmp/pa1-trace -- \
  "$MICROLLM_ENGINE_DIR/build/hip-release/benchmarks/microllm_bench_model" \
  --mode train --model tiny --device hip \
  --steps 10 --warmup 3 --batch 1 --context 8 --new-tokens 8
```

`MICROLLM_ENGINE_DIR` must point to a separate checkout of the `main` branch, as
described in the course root README.

## Submission template

```text
Environment and immutable command:
Correctness tolerance/result:
Warm-up/repetition method:
Baseline operator result:
Baseline end-to-end result:
Trace hotspot:
Hypothesis:
Minimal change:
Optimized operator result:
Optimized end-to-end result:
Peak engine memory:
Counterexample shape or setup cost:
Rejected/modified Agent suggestion and evidence:
Supported conclusion:
Unsupported conclusion:
```

The repository's own example is deliberately mixed: hipBLASLt speeds Model-S's
measured region but worsens five-token setup-inclusive throughput.

For a low-precision experiment, separately time quantization/scale computation and
GEMM. A pre-quantized FP8 matrix result is not an end-to-end model result. Compare
against the same hipBLASLt FP32 shape, not only the readable teaching kernel.
