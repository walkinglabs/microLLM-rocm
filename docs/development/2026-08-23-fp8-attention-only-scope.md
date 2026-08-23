# FP8 attention-only column scale scope

Exp145 identifies Attention as the best multi-Tensor reconstruction group in both official models;
Exp146 removes an inert output-head scope before this new candidate is introduced.

The experiment used `Fp8WeightScaleScope::AttentionOnly` to route Q/K/V/O projections to
output-channel weight scales.
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

Exp147 improves seven of eight Max/RMS metrics and keeps both T512 regressions below 5%, but Qwen
T512 RMS worsens 8.91%. Cross-model keep therefore fails. The scope remains only as a base for an
O-projection-only counterfactual. See [Experiment 147](../optimization-log/experiments/147-fp8-attention-only.md).

Exp148 then showed O-only preserves all DeepSeek improvements, removes the Qwen T512 regression and
uses one quarter of the post launches. The broader public enum/CLI/runner scope and focused tests
were removed. This document records historical behavior, not a current command.
