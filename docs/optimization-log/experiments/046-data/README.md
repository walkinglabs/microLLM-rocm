# Experiment 046 evidence

- `profile/kernel-stats.csv`: deterministic aggregation of the retained rocprofv3 Kernel trace.
- `profile/hip-api-stats.csv`: deterministic aggregation of the retained HIP API trace.
- `profile-summary.json`: scope, exact totals, category boundaries and next hypothesis.

The raw trace is about 22 MiB and is intentionally not committed. It can be regenerated
with `scripts/profile_hip.sh` and aggregated with `scripts/summarize_rocprof.py`. The
aggregates are small enough to review and the validator recomputes every total from them.

This is a profile-only experiment. It selects the next candidate but does not claim a
speedup and therefore does not add a row to `results.tsv`.
