# Step 19 — pointer-stable grouped gate/up model gate

Status: complete

## Evidence

- exact environment/shape registry; default has zero dispatches;
- one shared initialized kernel and 24/28 per-block device argument plans;
- 12 official model processes with top-1 and BF16 complete-logit gates;
- steady speedups 1.0176×/1.0117×;
- peak ratios 1.000008×/1.000003×;
- setup 57.0/56.8 ms and argument setup 0.58/0.70 ms;
- phase GEMM calls fall by exactly 24/28.

## Decision

Keep the explicit policy and infrastructure. Do not hard-code version-local indices or enable
unmeasured shapes by default.
