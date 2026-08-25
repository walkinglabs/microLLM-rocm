# One-process-per-GPU training bootstrap

Experiment 264 repeats the independent-rank tiny training path three times.
Each run starts rank1 first, atomically exchanges one opaque RCCL ID through a
fresh directory, launches rank0/rank1 on GPU0/GPU1, performs three updates, and
compares every parameter with a CPU global-batch reference.

- 6 rank processes and 18 rank-steps complete;
- all 728 values in 12 parameter Tensors are bit-exact across ranks;
- maximum CPU-reference difference is 1.19e-7;
- median rank-group initialization plus training is 5.27 seconds;
- injected invalid rank exits with code 1;
- the launcher terminates its peer blocked in RCCL init (code -15).

The bootstrap admits a rank-local bucket reducer. It currently all-reduces each
tiny parameter separately and does not yet migrate gradient-ready overlap or make
a performance/scaling claim.
