# Prefill Attention core diagnostics

Experiment 307 adds an opt-in numerical diagnostic path for scaled Q, QK scores,
causal-softmax probabilities, and P×V output. It is infrastructure evidence, not a
throughput benchmark.

The production path remains selected when value capture is absent, unfiltered, or
metadata-only. The diagnostic path preserves an additional T×T score tensor and must not
be used for timing.

Focused tests:

```bash
./build/cpu-debug/tests/microllm_tests \
  --gtest_filter='CpuOpsTest.CausalGqaAttentionMatchesComposedForwardAndBackward:TransformerModelTest.CachedPrefillTraceCapturesRequestedAttentionCoreWithoutChangingOutput'

./build/hip-release/tests/microllm_hip_tests \
  --gtest_filter='HipFullAttentionTest.CausalMhaGqaForwardBackwardMatchCpuWithoutTransfers'
```

The full gate results and explicit limitations are in [`verification.json`](verification.json).
The SVG is an autoresearch-style decision map: grey is the unchanged production route,
orange is diagnostic-only work, and green dots are completed evidence gates.

![diagnostics](diagnostics.svg)
