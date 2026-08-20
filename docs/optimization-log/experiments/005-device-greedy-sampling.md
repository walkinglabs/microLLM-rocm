# Experiment 005 — device greedy sampling

Status: `keep`

## Observed bottleneck

After Experiment 004, each greedy token still copies all 151,936 Qwen logits (or all
151,936 DeepSeek logits) to the CPU just to find one maximum index.

## Hypothesis

A deterministic block-parallel device argmax that returns one int32 token will remove
full-vocabulary D2H and improve both official generation rows without changing training.

## Scope

- allowed: FP32 device argmax and greedy/top-1 generator path;
- unchanged: stochastic top-k/temperature CPU reference, model, KV Cache, allocator,
  dtype and benchmark protocol;
- deterministic smallest-index tie rule; no global synchronization beyond reading the
  one selected scalar already required by the C++ token vector API.

## Required gates

- [x] hand values, equal maxima and non-finite sentinel
- [x] vocabulary 32/8192/151936
- [x] CPU/HIP exact index equality
- [x] selected token remains on GPU for the next embedding
- [x] only one int32 payload returns per generated token
- [x] exact Qwen/DeepSeek greedy tokens
- [x] stochastic sampling behavior unchanged
- [x] full CPU/sanitizer/HIP regressions

## Implementation

- one 256-thread block scans an arbitrary contiguous FP32 logits Tensor;
- each thread keeps a local best pair, then shared-memory reduction combines pairs;
- equal values choose the smallest flat index, matching `std::max_element`;
- any NaN/Inf produces int32 `-1`, which the generator turns into the existing error;
- device result shape is `[1,1]`, so it can be passed directly to the next cached forward;
- C++ copies only that int32 to append to the returned `std::vector`;
- stochastic temperature/top-k continues using the tested CPU algorithm.

## Correctness result

- CPU debug: `152/152` pass;
- ASan/UBSan: `150/150` pass;
- HIP release: `44/44` pass;
- Python/PyTorch operator parity: `4/4` pass;
- focused vocabularies `32/8192/151936` select exactly the CPU index;
- tie and non-finite contracts pass;
- focused four-token GPU generation transfers exactly two prompt int32 values H2D and
  four selected int32 values D2H; no generated token returns H2D;
- existing fixed-seed stochastic top-k tests remain unchanged;
- official Qwen and DeepSeek greedy sequences remain exact.

## Performance result

| Workload | Experiment 004 | Experiment 005 | Step speedup | PyTorch ratio | Peak memory ratio |
|---|---:|---:|---:|---:|---:|
| Qwen train | 72.328 token/s | 72.328 token/s | unchanged | 1.409241 | 0.896769 |
| Qwen generate | 85.645 token/s | 93.340 token/s | 1.09× | 1.329965 | 0.960651 |
| DeepSeek train | 49.474 token/s | 49.474 token/s | unchanged | 1.886431 | 0.797368 |
| DeepSeek generate | 35.788 token/s | 38.990 token/s | 1.09× | 0.624872 | 0.988733 |

```text
geometric score before  1.167931
geometric score after   1.219170
score improvement       4.4%
```

## Profiler result

Matched Qwen generation (`1` warm-up + `1` measured, four new tokens):

```text
profiled decode                44.04 → 50.04 token/s
D2H trace records                  9 → 1
hipMemcpy calls                   600 → 594
argmax Kernel calls                 0 → 8
argmax total Kernel time       0.000 → 2.038 ms
```

The single remaining D2H trace record is the app's separate full-forward logits report,
outside the generated-token loop. The runtime-level focused test supplies the byte proof:
exactly four bytes per generated token.

Raw JSONL and compact profiler tables are in [005-data](005-data/README.md). The direct
before profiler is Experiment 004's `after-*` data. Large PFTrace remains at
`/tmp/microllm-qwen-infer-profile-exp005/`.

## Replay command

```bash
python3 benchmarks/single_gpu/hf_model_matrix.py \
  --manifest build/hip-release/benchmarks/hf-models.local.json \
  --infer-binary build/hip-release/apps/microllm_hf_infer \
  --train-binary build/hip-release/apps/microllm_hf_train_step \
  --device hip --modes infer \
  --output /tmp/experiment-005-infer.jsonl
```

## Results

The greedy-path hypothesis is supported. Device stochastic top-k was intentionally not
implemented or claimed.

## Decision

`keep`. The next single variable is allocator churn.
