# Unique-gradient in-place accumulation evidence

Experiment 172 asks whether an Autograd gradient that has one persistent Storage owner can
receive its next dense contribution in place. The candidate is deliberately narrower than
the old generic copy-on-write attempt: diagnostics first prove the actual target operation,
shape and owner count on the retained T512 graph.

## Files

- `training.jsonl` and `summary.json`: allocating/in-place × Qwen/DeepSeek × three fresh
  processes, alternating policy order;
- `*-diagnostics.json`: one separate measured step per model/policy;
- `*-kernel-stats.csv` and `*-hip-api-stats.csv`: Qwen load plus three-step rocprofv3
  controls;
- `profile-summary.json`: exact call/time fields extracted from those CSV files;
- `coverage-summary.json`: fresh gcovr 8.3 source summary;
- `verification.json`: final build, correctness and regression gates.

## Result

| Model | Allocating | In-place | Speedup | Allocations saved / 2 steps | Peak |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 14,969.66 | 15,032.11 tok/s | 1.0042× | 144 | unchanged |
| DeepSeek Distill 1.5B | 6,267.24 | 6,236.90 tok/s | 0.9952× | 168 | unchanged |

The `1.01×` two-model gate fails. Qwen profiling explains why: 216 logical allocations
disappear across three steps, but backend allocation calls, HIP allocation/free calls,
all Kernel calls and add Kernel calls are identical. The exact-size cache had already made
these allocations device-API-free. The primitive and diagnostics remain available, while
engine and CLI defaults are false.
