# Experiment 098 data map

- `release-matrix/*.json`: eight candidate Release rows, including the preserved R8/S2 outlier.
- `paired/r8s2/*.json`: three alternating baseline/candidate pairs for the outlier.
- `paired/r8s4/*.json`: three pairs for the largest positive 4-slot shape.
- `paired/r4s4/*.json`: three pairs for the second 4-slot shape.
- `summary.json`: operator/model contracts, matrix and paired medians.
- `gates.json`: focused and complete correctness evidence.
- `environment.txt`: baseline/candidate, order and runtime contract.

The single matrix is not used to hide or average away a negative shape. The alternating data is the
decision gate and retains all 18 process outputs.
