# Experiment 100 data map

- `paired/r8s4/*.json`: three alternating baseline/candidate pairs plus candidate profile row.
- `paired/r8s2/*.json`: the matching higher-step evidence.
- `summary.json`: H2D mechanism and median performance.
- `gates.json`: focused/full correctness counts and keep decision.
- `environment.txt`: baseline, order, device and measurement contract.

The profile rows are ordinary continuous-only JSON without rocprof overhead. They exist to prove
transfer calls/bytes, while alternating pairs decide performance.
