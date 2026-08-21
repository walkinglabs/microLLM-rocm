# Experiment 106 data

- `raw.jsonl`: three compact B1/B2 pair records with all 31 stage error metrics.
- `summary.json`: stable per-stage error growth and complete-logit evidence.
- `environment.txt`: fixed trace/value boundary.
- `gates.json`: repository and focused checks.

Temporary full-value traces are intentionally not benchmark artifacts: the runner validates they
are complete, reduces them to reproducible error metrics, and removes them unless `--keep-traces`
is requested.
