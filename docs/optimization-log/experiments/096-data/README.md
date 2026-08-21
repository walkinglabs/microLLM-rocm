# Experiment 096 data map

- `summary.json`: state transitions, correctness contracts and five MI300X benchmark rows.
- `gates.json`: focused and full CPU/HIP/sanitizer evidence.
- `environment.txt`: build/device/measurement contract.
- `hip-benchmark/*.json`: untouched unspecified-build diagnostic records.
- `hip-release/*.json`: untouched Release records with 2 warmups and 10 repetitions.

The benchmark is deliberately a negative performance result. It proves the scheduler semantics and
shows why the divergent-row serial model oracle must be replaced before claiming serving speedup.
