# Experiment 006 — allocator audit before pooling

Status: `keep`

## Observed risk

rocprof sees thousands of `hipMalloc/hipFree` calls for the whole process, but that
includes checkpoint load, setup, full-forward reporting, warm-up and measured work.
The engine allocator records bytes but not calls, so it cannot isolate steady state.

Storage also does not know the last Stream that used its pointer. Reusing a released
pointer on the default Stream while an external Stream still reads it can create an
asynchronous use-after-free.

## Hypothesis

Resettable allocation/deallocation call counters over the measured interval will show
whether allocator churn justifies the Stream-ownership redesign required by safe reuse.

## Scope

- allowed: allocator call counters, JSON evidence, tests and design audit;
- unchanged: allocation backend, Tensor lifetime, model, operators, dtype and workload;
- no pointer caching, synchronization, memory-pool claim or throughput claim.

## Required gates

- [x] CPU and HIP counter lifecycle
- [x] Qwen/DeepSeek measured-interval call counts
- [x] unchanged numerical outputs and bounded memory
- [x] explicit safe/unsafe Stream ownership analysis
- [x] full regressions and async reuse stress

## Design evolution

### Audit-only baseline

The new counters showed that allocator churn was not just model load:

| Workload, five measured runs | Logical allocate/free calls |
|---|---:|
| Qwen generate | 12,345 / 12,345 |
| Qwen train | 9,200 / 9,200 |
| DeepSeek generate | 53,865 / 53,865 |
| DeepSeek train | 10,715 / 10,715 |

### Rejected candidate: enable before load/warm-up

It reduced backend calls and improved speed, but retained large model-load/full-forward
temporaries:

```text
Qwen inference reserved       about 3.96 GB
DeepSeek inference reserved   about 14.29 GB
```

This was rejected even though throughput improved.

### Kept candidate: measured steady state only

- cache is off by default;
- apps synchronize after warm-up, enable it, then reset measured counters;
- deallocation records a disabled-timing Event on the legacy default Stream;
- allocation reuses only exact-size blocks whose Event is ready;
- a runtime `Stream` or raw external `OpContext` permanently forbids reuse;
- disabled mode falls back to ordinary synchronizing `hipMalloc/hipFree`;
- active, cached and physically reserved bytes are reported separately;
- the pool is process-lifetime so it never calls HIP from uncertain static-destruction
  order; the driver reclaims retained blocks at process exit.

## Correctness result

- CPU debug: `152/152` pass;
- ASan/UBSan: `150/150` pass;
- HIP release: `53/53` pass;
- Python/PyTorch operator parity: `4/4` pass;
- exact-size reuse and non-default fallback tests pass;
- 256 asynchronous allocate/fill/release iterations plus exceptional destruction pass;
- external Stream/Event ordering and all existing graph/model tests pass;
- Qwen/DeepSeek tokens, final losses and parameter updates remain unchanged.

## Performance result

| Workload | Experiment 005 | Experiment 006 | Step speedup | PyTorch ratio | Logical peak ratio |
|---|---:|---:|---:|---:|---:|
| Qwen train | 72.328 token/s | 107.080 token/s | 1.48× | 2.086361 | 0.896769 |
| Qwen generate | 93.340 token/s | 134.868 token/s | 1.45× | 1.921682 | 0.960651 |
| DeepSeek train | 49.474 token/s | 69.770 token/s | 1.41× | 2.660319 | 0.797368 |
| DeepSeek generate | 38.990 token/s | 48.929 token/s | 1.25× | 0.784152 | 0.988733 |

```text
geometric score before  1.219170
geometric score after   1.700597
score improvement       39.5%
```

## Backend reuse and reserved memory

| Workload | Logical calls | Backend allocations | Reuses | Cached at report | Reserved at report |
|---|---:|---:|---:|---:|---:|
| Qwen generate | 12,345 | 305 | 12,040 | 2.26 MB | 1.978 GB |
| Qwen train | 9,200 | 1,154 | 8,046 | 1.108 GB | 9.013 GB |
| DeepSeek generate | 53,865 | 810 | 53,055 | 5.04 MB | 7.113 GB |
| DeepSeek train | 10,715 | 1,534 | 9,181 | 38.58 MB | 28.472 GB |

Generation exceeds the 10× backend-call gate. Training is only 7–8× and therefore
partially misses that line; this is not hidden by the improved aggregate score.

## Profiler caveat

The instrumented single Qwen decode changed `50.04 → 46.89 token/s`, while the fixed
five-run measurement changed `93.34 → 134.87 token/s`. rocprof magnifies the cost of
thousands of Event API calls: the new trace contains 2,760 `hipEventRecord` calls.
Both observations are retained. The keep decision uses the predefined uninstrumented
end-to-end metric; the profiler result warns that Event batching is future work.

Raw audit, rejected-candidate, final JSONL and compact profiler data are in
[006-data](006-data/README.md).

## Replay commands

```bash
python3 benchmarks/single_gpu/hf_model_matrix.py \
  --manifest build/hip-release/benchmarks/hf-models.local.json \
  --infer-binary build/hip-release/apps/microllm_hf_infer \
  --train-binary build/hip-release/apps/microllm_hf_train_step \
  --device hip --modes infer,train \
  --output /tmp/experiment-006.jsonl

./scripts/profile_hip.sh /tmp/experiment-006-profile -- \
  build/hip-release/apps/microllm_hf_infer \
  --config /tmp/qwen25-local/config.json \
  --weights /tmp/qwen25-0.5b-model.safetensors \
  --tokens 9707,1879 --device hip --new-tokens 4 \
  --warmup 1 --steps 1 --top-k 10
```

## Results

The measured allocator opportunity was real. The early-enable memory design was false;
the steady-state-only design is supported.

## Decision

`keep`, with the training 10× gate miss and profiler reversal explicitly retained.
