# Grouped weight-gradient capability evidence

Experiment 217 asks whether Q/K/V or gate/up weight gradients can share one FP32 hipBLASLt
GroupedGemm submission. The matrix covers two models, two projection families and two input
layouts.

| Layout | Qwen QKV | Qwen gate/up | DeepSeek QKV | DeepSeek gate/up |
|---|---:|---:|---:|---:|
| direct shared input (`N,T`) | 0 supported | 0 | 0 | 0 |
| one shared transpose + grouped (`N,N`) | 0 supported | 0 | 0 | 0 |

hipBLASLt inventories 8,153 direct and 9,172 materialized-layout algorithms, but
`isAlgoSupported` rejects every candidate for every official shape. This is a capability failure,
not a noisy performance loss. No Autograd/model route was added.

## Reproduce

```bash
HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/grouped_weight_gradient_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_grouped_weight_gradient \
  --output-directory /tmp/grouped-weight-gradient \
  --rows 512 --warmup 3 --repetitions 10 \
  --maximum-algorithms 16 --workspace-bytes 67108864
```

`raw.jsonl` contains all eight capability rows. `summary.json` records the discard decision.
`verification.json` records the complete release gates for the exact benchmark/runner source.
