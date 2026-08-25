# BF16 weight-gradient allocation attribution

Experiment 248 derives allocation identities from the retained 20-step model
records without restoring the rejected model route.

| Model | Routes | Allocation delta | Per route | Byte delta | Bytes/route |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 960 | 1,920 | 2 | 5,662,310,400 | 5,898,240 |
| DeepSeek Distill 1.5B | 1,120 | 2,240 | 2 | 12,037,652,480 | 10,747,904 |

For both models, bytes per route exactly equal the BF16 input cast+transpose
plus BF16 output-gradient cast. Backend allocation delta, peak-byte delta and
cached-byte delta are all zero. Cache-reuse delta equals allocation-call delta.

This proves source attribution but not a speed opportunity. A caller-owned
workspace would remove logical allocation/cache lookup only; it would not remove
the cast Kernels or GEMM. The next gate must measure allocating versus preallocated
wall and Event time before adding a public workspace contract.

