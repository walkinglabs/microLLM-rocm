# Step 21 — grouped rows 256/1024 capability

Status: complete

## Evidence

- 24 fresh processes and eight exact rows/model/projection cases;
- 64/64 candidates pass per process;
- device-arguments Event ratios span 1.124×–1.695×;
- QKV Max/RMS ≤0.000244/0.000109;
- gate/up Max/RMS ≤0.00000763/0.000000416;
- formal reinitialization medians are below 1.0 in all eight cases.

## Decision

Continue to cross-shape complete-model gating. Do not promote operator winner sets to defaults.
