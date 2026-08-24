# Caller-owned HIP Graph runtime evidence

Experiment 173 adds a move-only `HipGraphExecutable` around explicit-Stream capture,
instantiation and replay. The first supported boundary uses only caller-owned Tensor Storage;
every referenced pointer must outlive the executable and its last launch.

## Files

- `matrix.jsonl` and `summary.json`: 1/8/32/128/512 add nodes × 1/4096 elements ×
  eager/graph × three fresh processes with alternating order;
- `*-kernel-stats.csv` and `*-hip-api-stats.csv`: rocprofv3 128-node control;
- `profile-summary.json`: exact launch/API counts extracted from those CSV files;
- `coverage-summary.json`: fresh gcovr 8.3 source summary;
- `verification.json`: final build, regression and boundary gates.

## Result

| Add nodes | 1 element | 4,096 elements |
|---:|---:|---:|
| 1 | 0.595× | 0.590× |
| 8 | 0.890× | 0.827× |
| 32 | 1.328× | 1.207× |
| 128 | 1.567× | 1.503× |
| 512 | 1.909× | 1.728× |

Every one of the 60 processes is exact, transfer-free and reports `requested nodes + 1`
captured nodes. The stable failure is equally important: Graph setup/replay overhead loses for
one and eight tiny operations. It becomes useful only after enough submissions are grouped.

At 128 nodes, rocprofv3 records the same 2,583 executed Kernels. Eager issues 2,580
`hipLaunchKernel` calls; Graph issues 129 calls during capture plus 20 `hipGraphLaunch` calls
for replay. Total traced HIP API calls fall from 12,990 to 802.

This keeps the runtime primitive, not a model speed claim. Current model/autograd paths allocate
temporary Tensors and do not propagate an explicit Stream. The test proves synchronous allocation
inside capture is rejected and that the wrapper clears the sticky error before eager fallback.
