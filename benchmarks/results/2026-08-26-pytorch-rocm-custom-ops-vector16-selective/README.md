# Selective low-precision vector16 result

The accepted policy uses 16-byte packets only when all three pointers are 16-byte aligned, the
dtype is FP16/BF16, and the tensor has at least 4,194,304 elements. FP32, smaller tensors and
misaligned pointers retain the scalar kernel; the vector kernel handles an arbitrary final tail.

Six fresh processes again cover the same 20 PyTorch ROCm cases with five warmups and 25 measured
calls. Every output, gradient and loss is exact and allocator peaks remain equal. Compared directly
with the scalar microLLM matrix, the four low-precision 16M rows improve `1.277×–1.411×`; FP32
bandwidth rows pass the 0.95 non-regression gate. The final route still reaches only about
`0.816×–0.842×` native Torch, so this is a microLLM improvement rather than a Torch speedup claim.

- `raw.jsonl` and `summary.json`: final selective matrix;
- `comparison.json`: scalar/broad/selective complete comparison and admission decision.

