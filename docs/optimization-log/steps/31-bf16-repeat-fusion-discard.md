# Step 31 — BF16 cast+repeat fusion counterexample

Status: complete; model integration rejected

## Evidence

- CPU/HIP primitive is exactly equal to device cast then repeat;
- 48 fresh processes cover two model families and four sequence/batch cases;
- no timed payload transfers;
- only 3/8 rows pass 1.05; both B2 cases are neutral-negative;
- invalid host-cast pilot is rejected before timing evidence is accepted.

## Decision

Keep the explicit primitive, leave model/CLI/Auto unchanged.
