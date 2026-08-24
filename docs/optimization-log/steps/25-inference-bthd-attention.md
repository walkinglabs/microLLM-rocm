# Step 25 — inference BTHD Attention island

Status: complete

## Evidence

- 12 uninstrumented performance and 12 isolated diagnostic processes;
- strided calls 96/112→0;
- bytes 100.7/205.5 MB→0;
- speedups 1.1146×/1.0936×;
- peak decreases 4/7 MiB;
- complete logits are bit-exact.

## Decision

Keep explicit policy. Preserve old fallback for unmeasured RoPE/bias/cache/trace domains.
