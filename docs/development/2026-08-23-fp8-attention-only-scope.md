# FP8 attention-only column scale scope

Exp145 identifies Attention as the best multi-Tensor reconstruction group in both official models;
Exp146 removes an inert output-head scope before this new candidate is introduced.

`Fp8WeightScaleScope::AttentionOnly` routes Q/K/V/O projections to output-channel weight scales.
FFN and an optional LM head use device Tensor-amax. The scope is valid only with FP8
`OutputChannelAmax`; every other combination is rejected.

```text
--fp8-weight-scale-mode output-channel-amax
--fp8-weight-scale-scope attention-only
```

The policy name and JSON contain the scope. A tiny model prepares four column vectors and four
scalar weights when untied (three scalar weights when tied). Retained scale bytes are calculated
from actual vector lengths; preparation and graph-free output remain unchanged before/after.

The HIP gate proves four device column quantizations, four scalar preparations, no payload D2H and
zero hot-path weight requantization. It does not assume native outer-vector support: the cached
status-0 path still uses native scalar FP8 GEMM plus device post-scale.

Official same-revision device-Tensor control at T8/T512 decides precision and performance. The
candidate is not derived from the rejected output-head API and is not a default.
