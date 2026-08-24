# Packed weight-gradient evidence

Experiment 218 replaces two/three separate weight-gradient GEMMs with:

1. two/three real device-to-device 2D copies into one packed output-gradient Tensor;
2. one ordinary FP32 hipBLASLt GEMM;
3. shared packed output Storage that could be exposed as parameter-gradient views, so no split copy
   is hidden from the benchmark.

Three fresh processes per official case give:

| Model | Projection | Event speedup | Extra packed gradient/output |
|---|---|---:|---:|
| Qwen | QKV | 0.979× | 2.25 / 3.94 MiB |
| Qwen | gate/up | 0.835× | 19.0 / 33.25 MiB |
| DeepSeek | QKV | 0.897× | 4.0 / 12.0 MiB |
| DeepSeek | gate/up | 0.931× | 35.0 / 105.0 MiB |

All complete-output errors are at most `1.15484e-7`, but zero of four cases reaches the 1.05
operator gate. The route is discarded before Autograd/model integration.

## Reproduce

```bash
HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/packed_weight_gradient_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_packed_weight_gradient \
  --output-directory /tmp/packed-weight-gradient \
  --runs 3 --rows 512 --warmup 3 --repetitions 20
```

`verification.json` records the complete release gates for this exact benchmark and runner.
