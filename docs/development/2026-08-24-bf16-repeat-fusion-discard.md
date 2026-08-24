# BF16 V cast and repeat fusion counterexample

`repeat_interleave_bf16_to_float` combines a device BF16-to-FP32 conversion
with GQA head expansion. CPU and HIP outputs are exactly equal to the composed
device path.

The formal 48-process matrix is strongly positive for small B1 shapes, but only
3/8 cases clear 1.05 and both B2/T512 cases are neutral-negative. Model and CLI
routing are therefore cancelled. The explicit primitive, benchmark and failure
matrix remain available.

An initial benchmark accidentally used host-backed `Tensor::cast`; the transfer
gate detected 20 H2D and 20 D2H calls, invalidating that pilot before results
were accepted. See [Experiment 208](../optimization-log/experiments/208-bf16-repeat-fusion-discard.md).
