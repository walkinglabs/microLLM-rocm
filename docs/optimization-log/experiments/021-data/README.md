# Experiment 021 evidence

- `candidate-run-1.jsonl`: one complete four-workload official-model matrix;
- `after-hip-api-stats.csv`: DeepSeek candidate HIP API table;
- `after-kernel-stats.csv`: matching Kernel table;
- baseline profiler: Experiment 018 `after-hip-api-stats.csv`.

The candidate was stopped after one process because both generation workloads crossed
the 5% rejection gate and every workload moved down. The three-process rule prevents a
small claimed gain from being accepted; it does not require spending two more processes
to rescue a candidate that already fails the hard per-workload gate.

Candidate source was removed before this evidence commit.
