# Step 26 — BTHD sequence/batch matrix

Status: complete

## Evidence

- 36 performance and six isolated diagnostic processes;
- six speedups span 1.0852×–1.1421×;
- peak savings span 2–14 MiB;
- complete logits bit-exact; per-batch top-1 equal;
- Attention copies are zero in every case;
- B2 residual last-row copy is reported separately.

## Decision

Keep explicit measured cases. Cached-prefill and trace-value routes remain fallback-only.
