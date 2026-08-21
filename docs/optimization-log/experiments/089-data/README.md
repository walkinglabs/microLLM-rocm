# Experiment 089 data map

- `precision-summary.json`: DeepSeek T2048 B1/B8 complete cached-logit differences.
- `gates.json`: focused tests, official failure, identical-error rebuttal and rollback.
- `environment.txt`: frozen immediate-pool baseline and raw-packed candidate.

The result matches Experiment 088's two error records exactly. This rejects the explanation that
only the internal BF16 vector conversion caused the drift; the pair-loop code shape remains the
unresolved difference. No timing was accepted.
