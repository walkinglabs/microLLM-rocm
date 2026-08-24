# Step 20 — grouped policy composition

Status: complete

## Evidence

- 24 fresh processes across baseline/QKV/gate-up/both;
- both versus baseline 1.0655×/1.0474×;
- both versus QKV-only 1.0199×/1.0172×;
- both registries dispatch exactly once per block per forward;
- top-1 and BF16 complete-logit gates pass;
- peak ratios 1.00342×/1.00173×;
- combined setup 214.5/205.6 ms, with gate/up follow-up below 0.25 ms.

## Decision

Keep explicit composition for measured T512 environments. Defaults and unmeasured shapes remain
unchanged.
