# Experiment 053 — strided-batched hipBLASLt

## Missing framework capability

The optimized GEMM path accepted only rank-two Tensor inputs. Attention stores matrices as
`[batch, heads, rows, columns]`, so even a perfect library could not be selected without
copying or looping over heads.

## Implementation

The existing transpose-aware API now accepts equal-rank contiguous tensors with identical
batch dimensions. hipBLASLt layouts receive:

- `HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT`;
- `HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET` in elements;
- the same row-major-as-transposed-column-major mapping used by the validated 2D path.

The readable CPU fallback materializes only requested last-two-dimension transposes. Auto
dispatch is unchanged; Attention must explicitly pass the library gate first.

## Correctness

Tests cover rank 4 `[2,3,M,K]`, FP32/BF16, all four transpose combinations, output shape,
and zero payload transfer. A separate CPU test covers the readable batched fallback.

## Qwen T=512 shapes

| Layout | Readable | hipBLASLt | Valid speedup | Max error |
|---|---:|---:|---:|---:|
| `Q @ Kᵀ`, 14×512×64×512 | 4.752 ms | 0.181 ms | 26.23× | 5.36e-7 |
| `P @ V`, 14×512×512×64 | 4.398 ms | 0.0387 ms | 113.64× | 1.04e-6 |
| `dSᵀ @ Q` | invalid baseline | 0.0370 ms | excluded | 1.04e-6 |

![Strided-batched hipBLASLt](../assets/strided-batched-hipblaslt.svg)

The third readable measurement exposed a separate stream-dependency bug: temporary
`contiguous()` ran on the default stream while the benchmark GEMM used a non-default
stream. The hipBLASLt result is correct because it reads the original physical transpose.
No speedup is claimed for that row.

## Decision

Keep the reusable batched operator and benchmark. The next node integrates it only into
long-sequence Attention backward; end-to-end and memory gates remain independent.
