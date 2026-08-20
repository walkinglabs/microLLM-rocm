# Experiment 012 — two-stage large-vocabulary argmax

Status: `keep`

## Observed bottleneck

The deterministic device argmax uses one 256-thread block. At vocabulary 151,936 each
thread scans roughly 594 values; eight profiled calls consume about 2.04 ms.

## Hypothesis

For large inputs, multiple blocks can scan independent grid-stride partitions and write
small `(value,index,invalid)` partials; one final block combines them. Small inputs keep
the simpler one-block path. This should reduce greedy sampling latency without changing
tie or non-finite behavior.

## Scope

- FP32 flat argmax only;
- two-stage threshold 32,768 elements, at most 256 partial blocks;
- FP32 scratch stores exact indices (all accepted indices are below INT32_MAX; FP32 exact
  integer range covers current vocabularies);
- smallest index wins equal maxima; any non-finite value returns -1;
- generator/model/allocator/workload unchanged.

## Required gates

- [x] 32/8192 single-block and 151936 two-stage
- [x] equal maxima in different partial blocks
- [x] NaN/Inf in any partition
- [x] one int32 D2H contract
- [x] exact official tokens
- [x] three-process generation medians and profiler

## Implementation

- threshold `32768` keeps small inputs on the original one-block Kernel;
- large input launches up to 256 partial blocks with grid-stride scanning;
- each partial stores max value, exact FP32-encoded index and invalid count;
- one final block applies the same smallest-index tie rule;
- any partial containing NaN/Inf makes the result `-1`;
- scratch is device-local and reclaimed through the steady-state allocator.

## Correctness result

- CPU debug: `152/152` pass;
- ASan/UBSan: `150/150` pass;
- HIP release: `54/54` pass;
- Python/PyTorch operator parity: `4/4` pass;
- 32/8192/151936 paths pass;
- equal maxima at index 1 and the last index select 1 across partial blocks;
- a non-finite value in a distant partial returns `-1`;
- one-int32 D2H and next-token-on-device generation tests pass;
- official Qwen/DeepSeek token sequences remain exact.

## Three-process inference medians

Training is unchanged and reuses Experiment 009 medians.

| Model | Running-best median | Candidate samples | Candidate median | Change |
|---|---:|---|---:|---:|
| Qwen generation | 142.25 | 147.41 / 147.75 / 138.64 | 147.41 | +3.6% |
| DeepSeek generation | 53.04 | 53.36 / 55.91 / 53.14 | 53.36 | +0.6% |

```text
Qwen train ratio        2.086361
Qwen generate ratio     2.100335
DeepSeek train ratio    2.622345
DeepSeek generate ratio 0.855230
robust score            1.770568
previous score          1.752183
```

A post-candidate unmodified run measured 134.40/52.81 token/s, consistent with no hidden
candidate regression but also showing that the total gain is close to process noise.

## Profiler result

```text
single-block argmax            8 calls / 2.043 ms
partial + finalize             16 calls / 0.067 ms
argmax Kernel reduction                    96.7%
```

Instrumented whole decode changed `74.16 → 54.16 token/s`, opposite to the three-process
uninstrumented median. The extra Kernel and scratch/Event instrumentation amplify tool
overhead. Both facts are retained; keep is based on the prospective repeated-process
metric and absence of workload regression.

Raw repeats, post baseline and compact profiler tables are in
[012-data](012-data/README.md). Large PFTrace remains at
`/tmp/microllm-qwen-infer-profile-exp012/`.

## Results

Supported at Kernel level and by repeated generation medians. It does not claim a large
end-to-end improvement.

## Decision

`keep`, with the profiler reversal and marginal DeepSeek gain recorded.
