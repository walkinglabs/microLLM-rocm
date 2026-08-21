# 2026-08-21 — targeted FP32-layer prefix-pair retry discarded

The Experiment 068 candidate applied a paired prefix Kernel only to the strict mixed policy's one
FP32 Cache layer. A same-binary, same-GPU Release control removed 160 measured D2D calls and 167.8 MB
without changing logits, tokens, peak or decode throughput.

Cache preparation still regressed 1.53% and end-to-end generation regressed 0.59%. The route,
Kernel, API and tests were removed. Experiment 067's reference mixed policy remains the supported
strict option. See [Experiment 068](../optimization-log/experiments/068-targeted-prefix-pair-discard.md).
