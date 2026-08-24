# Scoped model Stream failure evidence

Experiment 175 tested a thread-local, nested `ScopedOpStream` that made every otherwise-default
`OpContext` use one explicit non-default Stream. Caller-owned operator and Graph tests passed, but
the first tiny Transformer correctness gate failed in all three repetitions.

`failure.json` records Max/RMS complete-logit differences:

```text
run 1  1.41198 / 0.474854
run 2  3.84635 / 0.931005
run 3  1.41198 / 0.474854
```

The failure is not a numerical tolerance issue. Model functions create temporary Tensor Storage;
their destructors run while queued non-default-Stream consumers are still asynchronous. A later
capture failure also leaves the next embedding launch reporting a previous capture error.

The scoped override and all candidate tests were removed. The explicit caller-owned Stream and
Graph APIs from Experiments 173–174 remain unchanged. Model Stream propagation now requires a
lifetime-aware deferred-release mechanism or planned activation arena first.
