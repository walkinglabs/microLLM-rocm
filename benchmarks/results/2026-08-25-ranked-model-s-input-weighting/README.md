# Ranked Model-S uneven-input weighting

Experiment 277 runs from clean revision `e43764b`. Rank0 uses B1T32 (32 valid
tokens) and rank1 uses B2T32 (64 valid tokens).

Equal-only mode exchanges counts and rejects both ranks before parameter
collectives. Token-weighted mode uses average tokens 48 and scales local mean
gradients by 0.666666687/1.333333373 before 57 RCCL averages.

The one-step result checks all 57 tensors and 15,586,176 values. Rank Max/RMS
are zero. Against the concatenated CPU B3T32 reference, parameter Max/RMS are
0.0077601/3.639e-6; row-weighted local loss differs by 3.20e-7.

The engine peak is 275,790,348 bytes for the larger local batch path. Timing is
reported only as smoke-resource evidence; the rank startup and first-use costs
dominate this one-step run.

Weighted ready-overlap is still unsupported. Synchronous weighted training is
retained; an overlap implementation would need to apply each rank's scale before
the corresponding ready bucket is packed and enqueued.
