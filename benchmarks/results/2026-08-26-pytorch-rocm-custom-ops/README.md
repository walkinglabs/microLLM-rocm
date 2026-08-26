# PyTorch ROCm Custom Op matrix

This package records the first native PyTorch ROCm dispatcher gate for the optional adapter.

- Torch `2.11.0+rocm7.13.0rc2`, HIP `7.13.99004`, `gfx942`;
- six fresh processes, alternating Torch-first and microLLM-first;
- five warmups and 25 measured calls per policy;
- 18 forward rows: add/multiply × FP32/FP16/BF16 × 4K/1M/16M elements;
- two FP32 Autograd branch rows at 64K and 1M elements.

`raw.jsonl` retains every worker. `summary.json` contains complete Max/RMS/loss checks,
Event/wall medians and PyTorch allocator peak deltas. Ratios are `Torch / microLLM`; below one
means the microLLM Custom Op is slower.

All 20 cases are exact and use the PyTorch-owned output allocation/current HIP Stream. Peaks are
equal. No speed median reaches one: FP32 16M reaches 0.933×–0.973×, while FP16/BF16 scalar typed
kernels are about 0.598×–0.638× at 16M. The integration is therefore admitted for compatibility,
not advertised as an elementwise performance win.

