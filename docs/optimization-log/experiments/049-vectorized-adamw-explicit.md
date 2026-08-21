# Experiment 049 — vectorized AdamW (explicit-only)

## Question

Experiment 048 showed that launch reduction did not move end-to-end time. Can one launch
move the large parameter/moment bytes more efficiently?

## Design

The readable Kernel keeps one element per thread. The candidate loads/stores four aligned
FP32 values per thread and writes four BF16 mirror values when present. It has:

- explicit `Scalar`, `Vectorized`, and `Auto` implementations;
- a 16-byte alignment gate and scalar tail;
- two-step parameter, first/second moment and BF16 mirror equality tests;
- an independent HIP Event benchmark with sampled numerical guard;
- CLI selection through `--adamw-implementation`.

## Exact-shape result

The shape set comes from Qwen/DeepSeek checkpoint parameter counts. With BF16 mirrors,
three rows clear a 5% operator gate:

| Elements | Scalar | float4 | Speedup |
|---:|---:|---:|---:|
| 802,816 | 0.0745 ms | 0.0624 ms | 1.194× |
| 136,134,656 | 11.223 ms | 10.628 ms | 1.056× |
| 233,373,696 | 19.406 ms | 17.685 ms | 1.097× |

But the two huge counts are embeddings/output heads in the real models and do **not** have
a BF16 mirror. Without that store, scalar is faster: `0.970×` and `0.980×` vector speedup.
Several middle counts are also neutral or slower.

![Vectorized AdamW explicit policy](../assets/vectorized-adamw-explicit.svg)

## Rejected sub-candidates

- Width 8 (two float4 values per thread) is slower on all twelve shapes; register pressure
  dominates additional instruction-level parallelism.
- `rsqrt` first produced NaN for zero gradient/second moment. After restoring the epsilon
  boundary, it passed numerical tests but remained slower than vector4 `sqrt`.

## Official-model pilot

Forcing Vectorized for every Qwen parameter produces speedups of
`0.967×、0.994×、0.965×、0.983×` on `1×3、2×3、1×32、1×128`; memory is unchanged. It fails
the model gate.

## Decision

Keep the implementation and benchmark as an explicit research path; keep `Auto` mapped to
Scalar. This makes operator optimization measurable and reproducible without silently
slowing real training. A future dispatch rule needs broader shape evidence and an official
model win before Auto may change.
