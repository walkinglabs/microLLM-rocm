# Experiment 020 evidence

- `offline-winner.txt`: explicit solution 293437, 10 cold + 100 measured iterations;
- `default-heuristic.txt`: default one-solution heuristic under the same protocol;
- `candidate-kernel-stats.csv`: framework micro-benchmark rocprof confirming the explicit
  Kernel was actually launched;
- `deepseek-run-{1,2,3}.jsonl`: official-model candidate processes with 2 warm-ups and
  5 measured iterations.

The initial `algo_method=all` sweep reported solution 293437 as its winner at 8.40 us,
but emitted more than a million lines and was not committed. The compact repeat is the
decision-quality evidence: explicit 293437 measured 9.50 us and the default heuristic
solution 293832 measured 9.87 us. The small difference did not survive the model gate.

Candidate code was removed before this evidence commit.
