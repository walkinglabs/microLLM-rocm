# FP8 output-column native outer-vector probe

The installed hipBLASLt header declares `HIPBLASLT_MATMUL_DESC_A_SCALE_MODE` with
`HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F`. In microLLM's row-major-to-column-major mapping,
this A-side vector corresponds to the user-visible weight output columns.

The output-column path now submits that mode once. Success sets `output_column_native_status=1` and
returns the library result without a post kernel. `INVALID_VALUE`, `INTERNAL_ERROR`, or
`NOT_SUPPORTED` sets status 0 and immediately retries the proven scalar-scale plus device post-scale
path. The result is cached per thread, so later Linear calls never repeat a known failed submission.

The MI300X 128x128 E4M3-FNUZ probe records status 0 and one post-scale call. It still has one native
scalar FP8 shape, zero software fallback and zero hot-path payload transfers. Therefore the current
header exposes the API but the installed runtime does not execute it for this FP8 boundary.

The CLI and official runner carry the tri-state field. This closes the direct outer-vector speed
idea on the measured stack; it does not delete the portable attempt because another architecture or
library version may return status 1.
