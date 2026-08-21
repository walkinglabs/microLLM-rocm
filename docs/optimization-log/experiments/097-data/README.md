# Experiment 097 data map

- `release-matrix/*.json`: eight candidate Release rows matching Experiment 096 shapes.
- `paired/r4s4/*.json`: three alternating baseline/candidate pairs at requests 4, slots 4.
- `paired/r8s2/*.json`: three alternating pairs at requests 8, slots 2.
- `summary.json`: mechanism, full matrix and paired medians.
- `gates.json`: focused/full correctness counts and keep decision.
- `environment.txt`: frozen baseline/candidate and measurement contract.

Uniform rows are a no-regression control. Their implementation path is unchanged, so their
cross-process increase is not attributed to active-row compaction.
