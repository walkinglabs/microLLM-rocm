# FP16/BF16 caller-owned Softmax baseline

Six fresh MI300X processes cover FP16/BF16 at widths 1/17/128/1024/4096. The new path reduces in
FP32 and rounds directly to the low-precision caller output; it never allocates a Tensor-shaped
FP32 temporary. All ten PyTorch oracle rows pass, every pointer is identical/non-owning, and
microLLM measured peak extra is zero.

This is not a performance-ready Kernel. The readable baseline assigns one thread per row and scans
the whole width serially. Torch/microLLM Event is about 0.08–0.10× at width128, 0.011× at width1024,
and 0.0036–0.0040× at width4096. Width1 is faster, which confirms launch/row work rather than
wrapper copying.

The capability is kept as a CPU/HIP/reference seam. A block-parallel reduction is required before
any model or performance route. `raw.jsonl` and `summary.json` retain all measurements.
