# Step 15 — grouped-QKV prewarm

Status: complete

## Evidence

- zero-warmup, one-prefill, 18 fresh processes;
- lazy first request: 5744/5741 ms;
- explicit prewarm: 915/886 ms;
- first admitted request after prewarm: 4852/4795 ms;
- complete logits remain inside BF16 gates;
- repeated same-row prewarm is a no-op.

## Decision

Keep explicit prewarm for serving admission. Do not claim reduced total startup and do not change
one-shot defaults.
