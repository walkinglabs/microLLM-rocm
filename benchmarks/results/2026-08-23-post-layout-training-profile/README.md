# Post-layout training phase profile

Experiment 165 repeats the load+three-step minus load+one-step profile after complete
Attention context-layout fusion.

- `one-step-kernel-stats.csv`: load plus one training step;
- `three-step-kernel-stats.csv`: load plus one warm-up and two measured steps;
- `profile-delta.json`: exact-name `(three-step - one-step) / 2` categories;
- `verification.json`: hotspot closure and next-candidate decision.

Derived Kernel time is 33.349 ms/step. Strided materialization has no positive delta and
is absent from categories. GEMM is 56.55%; AdamW is 16.76%. Existing exact-solution and
AdamW implementation searches are already closed by model evidence.

The next open boundary is host-side setup for the new interleaved Attention GEMMs. Qwen
executes P×V/dP/dV 72 times per step and DeepSeek 84. Each current call constructs three
hipBLASLt layouts and one description. Experiment 163's wall-minus-Event medians imply
about 0.67/0.74 ms setup exposure per model step; this is a hypothesis, not yet a gain.
