# 2026-08-25 — rejected scoped Autograd producer cleanup

Experiment 261 closes the promotion from independently fast operator to Autograd. The cleanup
removes all generic/zero/overwrite target APIs, node state, dispatch controls/counters, the
backward benchmark and matrix runner, and seven route-specific CPU/HIP tests.

The retained surface is deliberately smaller:

- ordinary first leaf contribution assigns the producer Tensor;
- later contributions use the existing accumulation rules;
- `matmul_weight_gradient_out_` remains a public caller-owned operator with CPU/HIP/PyTorch tests;
- its independent five-shape benchmark and Experiment 260 evidence remain reproducible.

The log validator requires every target/dispatch symbol to remain absent while requiring the out
operator. Post-cleanup validation passes CPU `360/360`, ASan/UBSan `358/358`, RCCL `30/30`, and
119 registered native/Python test sources.
