# Per-device hipBLASLt handle evidence

Experiment 178 fixes a pre-existing multi-GPU correctness failure. A process-wide static
hipBLASLt handle was created while GPU 0 was current, then reused after rank-local execution
switched to GPU 1. Vendor GEMMs failed with `invalid device ordinal`; pure RCCL collectives were
unaffected.

The retained implementation uses one thread-local handle per device index. BF16 plan and
Attention layout caches were already keyed by device; they now receive the matching handle.

Correctness gates:

- the five previously failing RCCL model/CLI tests pass;
- the complete RCCL set improves from 6/11 to 11/11, plus 2/2 package consumers;
- a direct `GPU0 → GPU1 → GPU0 → GPU1` FP32/BF16 GEMM test is exact;
- the BF16 cache records two device entries, two misses and two later hits.

Single-GPU non-regression uses the final Experiment 177 legacy rows as the previous-revision
baseline. Three fresh current processes per workload give:

| Workload | Current / previous |
|---|---:|
| Qwen T512 inference | 1.023× |
| Qwen T512 training | 1.000× |
| DeepSeek T512 inference | 0.998× |
| DeepSeek T512 training | 1.001× |

All output fingerprints, losses and observed parameter updates are exact. The minimum ratio is
0.9979, above the declared 0.95 non-regression gate.

Files:

- `raw.jsonl`: 12 fresh current processes;
- `summary.json`: four previous/current median comparisons;
- `verification.json`: exact same-revision test matrix.
