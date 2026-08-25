# Ranked persistent-bucket matrix

Experiment 268 compares per-parameter, transient-bucket and persistent-bucket
reducers from clean revision `5b6e78f`. Each policy has three fresh two-rank
process groups; step 1 is cold and steps 2–3 form six steady samples.

| Policy | Steady reducer | Steady step | Backend alloc/steady step | Current/rank | Peak/rank |
|---|---:|---:|---:|---:|---:|
| per parameter | 2.692 ms | 8.712 ms | 0 | 249.38 MB | 262.58 MB |
| transient bucket | 4.440 ms | 10.311 ms | 60 | 249.38 MB | 314.90 MB |
| persistent bucket | 2.886 ms | 8.251 ms | 0 | 311.72 MB | 387.27 MB |

Persistence improves reducer/complete-step time by 1.539×/1.250× versus the
transient bucket and changes later backend allocations from 60 to zero. Versus
per-parameter, complete step is 1.056× faster, but the reducer itself is still
0.933× and therefore takes 7.2% longer.

The plan owns 124,689,408 bytes per rank. Persistent current memory is 62,344,704
bytes above both controls; peak is 124,689,408 bytes above per-parameter and
72,376,320 bytes above transient. Pack and unpack remain 57 copies each per step.

All 15,586,176 parameter values remain rank-exact after three AdamW steps. CPU
Max/RMS are 0.0062715/3.701e-6, mean loss difference is at most 1.967e-5, and
peer failure remains bounded. Temporary weight and communicator files are removed.

Persistent copies are kept as an explicit counterfactual, not enabled by default.
The next experiment makes parameter gradients views into persistent buckets,
removing independent unpack storage and 57 unpack copies before any overlap work.
