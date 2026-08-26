# Rejected broad wave-reduction typed Softmax

The candidate replaces both 256-thread shared-memory trees in the cached width4096
Kernel with intra-wave shuffles and one cross-wave combine. The same six-process,
two-order FP16/BF16 PyTorch matrix passes every correctness, pointer and zero-peak-extra
gate.

Relative to the retained cached baseline:

| dtype | Event gain | wall gain | 1.05× Event+wall gate |
|---|---:|---:|---:|
| BF16 | 1.050× | 1.033× | fail |
| FP16 | 1.071× | 1.070× | pass |

The broad route is rejected because both dtypes were in scope and BF16 wall time misses
the declared gate. The wave helpers and dispatch were removed. This result does not
authorize averaging the dtypes or silently enabling the candidate only because FP16
passes; an FP16-only policy requires its own explicit experiment.

`raw.jsonl` and `summary.json` retain the complete rejected measurement.
