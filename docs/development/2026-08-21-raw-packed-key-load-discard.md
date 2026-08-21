# Raw-packed BF16 Key load rejection

Experiment 089 replaces the rejected internal BF16 vector reinterpretation with one raw 32-bit
load and explicit reconstruction of two public `hip_bfloat16` scalar values. Focused tests pass,
but the official T2048 B1/B8 errors and token behavior exactly match Experiment 088.

The internal-vector-only explanation is rejected. The local pair-loop search is closed until a
position-dot intermediate or compiler-codegen gate can prove equivalence. The candidate is fully
reverted and no timing is accepted.

See [Experiment 089](../optimization-log/experiments/089-raw-packed-key-load-discard.md).
