# Ranked Model-S cold/steady reducer matrix

Experiment 267 runs three fresh process groups per policy from clean revision
`ff2be89`. Every group trains Model-S for three `B1×T32/rank` steps. Step 1 is
retained as cold; steps 2–3 produce six steady samples per policy.

| Policy | Cold reducer median | Steady reducer median | Steady CV | Steady step median |
|---|---:|---:|---:|---:|
| per parameter | 53.93 ms | 2.837 ms | 27.74% | 8.864 ms |
| transient 25 MiB bucket | 40.82 ms | 4.205 ms | 2.72% | 10.396 ms |

Cold-only timing says bucket is 1.321× faster. Steady timing reverses the result:
bucket reducer time is 1.482× larger and complete step time is 1.173× larger. The
bucket's low steady CV makes this a stable counterexample, not a single noisy run.

Every steady bucket step creates 60 logical and backend allocations, allocates
124,689,408 bytes, performs 57 pack copies and 57 unpack copies, then deallocates 60
objects. Per-parameter reduction performs none of these engine allocations or copies.

All 57 tensors and 15,586,176 values remain rank-exact after three AdamW steps.
Rank/CPU Max and RMS are 0.0062715 and 3.701e-6; mean-rank/global-batch loss differs
by at most 1.967e-5. Peer failure remains bounded. Temporary weight and communicator
files are removed.

The transient bucket remains a correctness implementation but is rejected as the
steady performance route. The next counterfactual reuses rank-local bucket and unpack
storage to remove the measured 60 backend allocations before considering overlap.
