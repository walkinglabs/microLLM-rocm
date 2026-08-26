# SwiGLU zero-stride scalar-seed Autograd route

PyTorch `sum()` sends a shape-sized gradient view with every stride zero and only four bytes of
storage. The previous bridge called `.contiguous()`, materializing one full FP32 gradient before
the fused backward Kernel. The retained route detects only this exact layout and passes a
one-element device view to `swiglu_backward_scalar_seed`; mean, weighted and general gradients
keep the ordinary contiguous contract.

The same six-process, 15-case matrix remains within every dtype-specific precision gate. For FP32
forward+backward, measured microLLM peak falls from 263,680 to 1,536 bytes at 64K and from
4,195,840 to 1,536 bytes at 1M. Event time is non-regressing/slightly better, but the final path is
still only about `0.773×–0.781×` native Torch. The route is kept for its proven memory removal;
the hypothesis that materialization alone explains the full training gap is rejected.

- `raw.jsonl`, `summary.json`: post-route matrix;
- `comparison.json`: exact pre/post timing, peak and admission gates.

