# Activation arena evidence

Experiment 180 allocates one stable backing region outside capture/replay and follows a known
two-slot liveness plan. It compares deferred temporary allocation, eager arena submission and
arena HIP Graph replay.

The formal matrix covers 8/32/128/512 nodes, 1/4096 elements, three fresh processes per mode and
20 timed repetitions: 72 processes.

| Nodes | Eager arena speedup E1 / E4096 | Arena Graph speedup E1 / E4096 | Graph break-even replays E1 / E4096 |
|---:|---:|---:|---:|
| 8 | 1.071× / 1.072× | 1.315× / 1.314× | 1,280 / 1,174 |
| 32 | 1.369× / 1.359× | 2.187× / 2.047× | 171 / 173 |
| 128 | 1.480× / 1.455× | 2.668× / 2.561× | 40 / 40 |
| 512 | 1.595× / 1.768× | 2.953× / 3.066× | 10 / 9 |

All outputs are exact. Both arena paths use two addresses. Graph contains exactly `N+1` Kernel
nodes and no allocation/free nodes. Backing capacity is 512 bytes for one-element rows and
32,768 bytes for 4,096-element rows.

The representative profile keeps 2,971 executed Kernels in every mode. Eager arena reduces
whole-process synchronous malloc/free from 2,948/2,948 to 5/5. Graph keeps 5/5, reduces host
Kernel launches from 2,967 to 129, and adds 23 Graph launches.

Files: `raw.jsonl`, `summary.json`, `profile-summary.json`, profiler CSV tables and
`verification.json`.
