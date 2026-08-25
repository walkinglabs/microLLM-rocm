# 2026-08-25 — one-process-per-GPU bootstrap infrastructure

## Components

- `RankCommunicator`: one RCCL rank, local GPU and communication Stream;
- opaque communicator ID generation and exact-byte validation;
- rank worker with one tiny model, one AdamW, local batch and averaged gradients;
- CPU global-batch reference mode in the same executable;
- atomic rank0 ID-file publication and bounded peer wait;
- Python group launcher that starts peers, applies a total deadline, terminates survivors after
  any peer failure, and compares all parameter values;
- repeated-matrix runner plus normal and peer-failure CTest paths.

## Pilot evidence

Two independent OS processes run three steps on GPU0/GPU1. All 728 values in 12 parameter Tensors
are bit-exact across ranks and within 1.19e-7 of the CPU global-batch reference. In the injected
failure, invalid rank 2 exits and the launcher terminates rank0 while it waits inside RCCL init.

The worker currently all-reduces each small gradient in registration order and is correctness
only. Formal repeated launch evidence comes before bucketization, checkpoint ownership, or
gradient-ready overlap migration.

Infrastructure gates pass RCCL-labelled `41/41`; the machine audit registers 42 graph API entries
and 123 native/Python test sources.
