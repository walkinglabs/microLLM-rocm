# Step 22 — grouped sequence/batch model matrix

Status: complete

## Evidence

- 36 fresh model processes over six model/case pairs;
- speedups span 1.0212×–1.1075×;
- batch-row top-1 and BF16 complete-logit gates pass;
- peak ratios span 1.00088×–1.00657×;
- combined setup stays 208.1–212.2 ms;
- real CLI fixture covers B1, B2-last and B2-full file export.

## Decision

Keep explicit rows256/1024 policies. Preserve batch and sequence as separate workload identity.
