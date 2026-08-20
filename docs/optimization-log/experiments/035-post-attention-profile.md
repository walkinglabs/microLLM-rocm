# Experiment 035 — profile after BF16 Attention retention

Status: `profile handoff` — host plan construction selected

## Before and after Kernel timeline

| Category | Exp033 BF16 FFN | Exp035 BF16 FFN+Attention |
|---|---:|---:|
| total Kernel time | 237.60 ms | 109.20 ms |
| GEMM time | 160.71 ms | 70.47 ms |
| GEMM share | 67.64% | 64.54% |
| cached Attention | 24.28 ms | 8.49 ms |
| BF16 cast time | 14.63 ms | 10.45 ms |
| dispatches | 10,038 | 11,214 |

The lower Kernel times confirm that BF16 Attention reaches different hipBLASLt kernels.
The extra 1,176 dispatches come from Attention input/output casts and the additional
one-time preparation casts.

## What Kernel trace cannot see

Every BF16 GEMM still built and destroyed one hipBLASLt operation description plus three
matrix layouts on the host. That work is outside GPU Kernel duration. With 3,743 GEMMs in
the trace, repeated immutable descriptor construction became the next hypothesis.

The candidate is deliberately narrower than discarded Experiment 007:

- only the new BF16 mixed/output path is cached;
- exact key is `(M,K,N,output dtype)`;
- cache is thread-local and immutable;
- algorithms, workspace, FP32 descriptors and FP8 scale-pointer descriptors are untouched;
- direct BF16→FP32 capability lookup also becomes thread-local rather than locking globally.

Experiment 007 remains a valid FP32 failure. A new workload/path requires new evidence,
not retroactive reinterpretation.
