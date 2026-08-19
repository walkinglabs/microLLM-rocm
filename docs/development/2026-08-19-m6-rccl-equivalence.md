# 2026-08-19 — M6 RCCL two-GPU equivalence baseline

## Contract

Before overlap or scaling claims, prove that two ranks average gradients and update
the same model as a single rank processing the equivalent global batch. Validate all
rank-local tensors before entering a collective and abort communicators on collective
failure.

## Implementation

- single-process `ncclCommInitAll` communicator clique;
- one non-blocking HIP Stream per rank;
- grouped in-place float32 sum all-reduce;
- optional post-reduction averaging;
- device/order/shape/dtype/contiguity validation before group start;
- partial-initialization cleanup, grouped-call cleanup, explicit abort state;
- `Value::set_grad` with shape/device/dtype validation for distributed gradients.

## Hardware/software

- four visible AMD Instinct MI300X VF (`gfx942`);
- GPUs 0–3 share NUMA node 0;
- every pair reports one hop and XGMI link;
- RCCL/NCCL API version macros: 2.28.3;
- two-rank test uses GPUs 0 and 1.

## Evidence

1. rank values `[1,2,3,4]` and `[3,4,5,6]` average to `[2,3,4,5]` identically;
2. mismatched shapes fail before communication and do not abort a healthy communicator;
3. two tiny Transformer ranks see different local B1 batches, average every parameter
   gradient, and take AdamW step one;
4. a CPU single-rank B2 global-batch reference takes the equivalent step.

Observed maximum differences:

```text
rank_parameter_max_difference=0
single_vs_two_rank_max_difference=1.49012e-08
```

All three RCCL-labelled tests pass. Communicator initialization dominates their
roughly 3–4 second individual runtime; no scaling conclusion is made.

## Next step

The baseline communicates each parameter independently and synchronizes each call.
Gradient buckets must reduce collective count while preserving these exact-equivalence
tests. Only then should communication/computation overlap be measured.
