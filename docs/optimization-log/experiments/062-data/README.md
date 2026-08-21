# Experiment 062 evidence

- `formal/`: Qwen/DeepSeek × context 8/512/1024 × two frameworks × three fresh
  processes, cached-only, one warm-up and two measured iterations.
- `long/`: context 2048 one-process extension.
- `profile-token/`: explicit legacy token-replay Qwen T512 control.
- `profile-full/`: full-sequence prefill-to-cache Qwen T512 candidate.
- `comparison.json`: cache prepare/end-to-end, memory and failure-boundary contract.
- `profile-summary.json`: exact Kernel/HIP API aggregate and call-count reduction.

The first implementation copied packed heads into a capacity-strided Storage as one
contiguous buffer. Continuation logits failed. The final implementation copies each head
into its capacity stride and returns only last-token logits; all committed measurements
were recorded after both fixes.
