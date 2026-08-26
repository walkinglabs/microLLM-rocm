# FP16 typed Softmax submission attribution

Six fresh C++ processes alternate raw-first and C++-first order. Each process measures
seven samples of 25 width4096 launches on one explicit Stream. The committed Python
C API and PyTorch numbers come from the same 1024-thread six-process matrix.

| layer | Event median | ratio to previous layer |
|---|---:|---:|
| PyTorch | 4.530 μs | reference |
| raw microLLM launcher | 4.764 μs | 1.052× PyTorch time |
| C++ `softmax_typed_out_` | 4.815 μs | 1.011× raw |
| Python/C API | 5.086 μs | 1.056× C++ |

All raw/C++ outputs have maximum error 5.96e-8 and timed payload-transfer calls are
zero. The total Python/PyTorch Event-time ratio is 1.123×. This splits the remaining
gap: about 0.23 μs is below the public C++ API, about 0.05 μs is C++ validation, and
about 0.27 μs appears between C++ submission and repeated Python/C API launches.

`raw.jsonl` stores every C++ process. `summary.json` stores the joined medians and
ratios. The result does not claim those deltas are perfectly additive across separate
process families; it bounds the next work scale.
