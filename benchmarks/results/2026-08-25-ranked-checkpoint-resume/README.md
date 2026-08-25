# Ranked checkpoint ownership and resume

Experiment 272 runs from clean revision `8357bcf`:

```text
two ranks × 2 steps -> rank0 interrupted checkpoint
new two-rank group resumes -> 3 steps -> resumed-final checkpoint
fresh two-rank group -> uninterrupted 5 steps -> uninterrupted-final checkpoint
```

All three checkpoint files are 10,796 bytes. Resumed-final and uninterrupted-final
are byte-for-byte equal, covering model parameters, both AdamW moment collections,
optimizer step and ExperimentState. Rank/rank and resumed/uninterrupted parameter
differences are zero.

Only rank0 reports a write: three successful groups produce three rank0 writes and
zero nonzero-rank writes. In the failure injection, rank0 explicitly returns 1 before
publishing its ready marker; the launcher terminates the waiting peer with -15.

Checkpoint, ready, temporary and communicator files are deleted after verification.
The result is a correctness/reliability gate, not a performance claim. It admits a
separate Model-S checkpoint size/runtime smoke.
