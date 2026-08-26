# Broad vector16 counterexample

This is the rejected first vectorization matrix. It routed every aligned FP32/FP16/BF16
add/multiply call through 16-byte packets.

All 20 PyTorch ROCm cases remained exact with equal allocator peaks. FP16/BF16 16M improved
materially, but FP32 16M became only `0.845×–0.879×` as fast as the retained scalar microLLM
kernel. Small and medium rows had no stable gain. A universal vector route is therefore rejected.

The raw and summary schemas match the scalar baseline. The accepted selective rebuttal lives in
the sibling `2026-08-26-pytorch-rocm-custom-ops-vector16-selective` directory.

