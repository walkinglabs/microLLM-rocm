# PyTorch ROCm fused SwiGLU Custom Op

This evidence package compares `torch.ops.microllm.swiglu(gate, up)` with
`torch.nn.functional.silu(gate) * up` on Torch `2.11.0+rocm7.13.0rc2`, HIP
`7.13.99004`, `gfx942`.

Six fresh processes alternate Torch-first and microLLM-first. Each policy receives five warmups
and 25 measured calls. The 15 cases cover FP32/FP16/BF16 forward at 4K/1M/16M elements and
forward+backward at 64K/1M. Max/RMS/loss tolerances are explicit per dtype.

All precision gates pass. At 16M, fused forward is `1.570×` FP32, `1.178×` FP16 and `1.142×`
BF16 by Event; PyTorch allocator measured peak is cut in half because the SiLU intermediate
disappears. Small/medium rows are not universal wins. Forward+backward is only
`0.597×–0.761×`; low-precision fallback formulas also use more temporary memory.

The public conclusion is therefore bounded: keep the optional fused operator and its large-forward
evidence, but do not claim training acceleration. `raw.jsonl` retains every process and
`summary.json` contains the complete matrix.

