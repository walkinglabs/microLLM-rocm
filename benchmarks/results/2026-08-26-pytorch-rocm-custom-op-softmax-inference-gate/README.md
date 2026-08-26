# C++ PyTorch Custom Op Softmax with inference gate

The C++ Autograd registration now calls `Function::apply` only when GradMode is enabled
and the input requires gradients. Otherwise it dispatches directly to the engine forward
on PyTorch's current HIP Stream. CPU, Meta/fullgraph and C++ backward remain available.

Six fresh MI300X processes cover FP16/BF16 widths 1/17/128/1024/4096 in both call
orders. All 10 precision and distinct-output gates pass; Custom Op peak bytes equal
native Torch in every row.

- FP16 width4096 improves 6.640→5.732 μs, or 1.158×, versus the initial adapter;
- width1024 reaches 1.026× native for FP16 and 0.993× for BF16;
- width4096 remains 0.795× native for FP16 and 0.529× for BF16.

The functional adapter is retained for ecosystem integration, not promoted as a
universal performance winner. A caller-owned Custom Op variant would require a separate
aliasing/mutation schema and benchmark.
