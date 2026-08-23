# FP8 output-column weight scale operator

Exp141 identifies weight rounding as the dominant Qwen source, while Exp142 shows that replacing
native GEMM is not a precision fix. The new operator boundary therefore improves weight granularity
without abandoning native FP8 compute.

`quantize_fp8_columns_dynamic` accepts a contiguous rank-two `[K,N]` weight, computes one
device-resident scale per output column, and returns `Fp8ScaleMode::OuterColumn`. CPU and HIP
dequantization both map element `(k,n)` to `fp8(k,n) * scale[n]`.

The installed MI300X runtime rejects hipBLASLt outer-vector FP8 scaling. The native workaround uses
`scale[0]` as the library scalar, then launches one device kernel multiplying output column `n` by
`scale[n] / scale[0]`. Algebraically:

```text
GEMM gives sum(Aq * Wq) * activation_scale * scale[0]
post-scale gives the above * scale[n] / scale[0]
= sum(Aq * Wq) * activation_scale * scale[n]
```

No scale or payload returns to the host. The dispatch counter separates native GEMM from the output
column-scale launch. A 128x128 MI300 test proves native dispatch, one post-scale call, zero fallback,
zero hot-path H2D/D2H and agreement with the FP32 matrix reference. CPU tests cover independent
column ranges, dequantization, matmul and invalid rank.

The model now exposes this operator through an opt-in preparation policy, but official-model evidence
remains a separate node; operator correctness is not yet a claim that per-column weights improve an
LLM.
