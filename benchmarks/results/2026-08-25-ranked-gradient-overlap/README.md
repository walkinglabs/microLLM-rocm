# Ranked gradient-ready overlap matrix

Experiment 270 compares five reducers from clean revision `3320c43`. Each policy
has three fresh two-rank groups; step 1 is synchronous plan warmup and steps 2–3
form six steady samples.

| Policy | F/B or enqueue | Reducer finish | Complete step | Current | Peak |
|---|---:|---:|---:|---:|---:|
| synchronous views | 5.257 ms | 3.080 ms | 8.195 ms | 249.38 MB | 324.93 MB |
| overlap views | 6.456 ms | 1.413 ms | 8.152 ms | 249.38 MB | 324.93 MB |

Overlap makes the optimizer-side finish wait 2.180× faster, but hook/Event/pack/
collective enqueue adds 1.199 ms to the backward interval. Complete steady step
improves only 1.0052×, below the fixed 1.01 gate. Synchronous/overlap total CV is
4.10%/3.47%, and their six-sample distributions overlap.

Steps 2–3 enqueue all three buckets in fixed order on every rank. Later backend
allocations remain zero; plan capacity, current and peak are exactly equal to
synchronous views. All 15,586,176 values remain rank-exact, CPU and loss gates
pass, and peer failure remains bounded.

The overlap implementation remains an explicit teaching/research route, not a
default or a performance claim for Model-S T32. This fixed workload's ranked
reducer local search is closed. A future overlap claim must change workload scale
or communication topology and establish a separate comparison track.
