# Deferred HIP deallocation scope

The new explicit scope retains same-thread/same-device allocations until one non-default Stream
finishes, then releases them together. It is fixed-capacity, non-nestable and independently
reports pending/total blocks and bytes.

MI300X tests and 36 fresh benchmark processes are exact and transfer-free. Compared with the
correctness-preserving per-node synchronization control, wall speed improves `2.28×–2.74×`.
Profiler reduces Stream synchronizations 320→10 while keeping 323 Kernels and 322 allocations/
frees unchanged.

The tradeoff is physical residency: 128 nodes ×4096 elements retain 2,080,768 bytes until finish.
This primitive does not route operators or enable Graph capture. The next experiment must combine
it with explicit model Stream routing and re-run complete logits before any performance claim.

Full report: [Experiment 176](../optimization-log/experiments/176-deferred-hip-deallocation.md).
