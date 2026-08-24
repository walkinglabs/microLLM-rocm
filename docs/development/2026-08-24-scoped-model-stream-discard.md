# Scoped model Stream candidate removed

A thread-local nested Stream override passed caller-owned operator tests but failed the first
complete tiny-Transformer logit gate in three runs. Max/RMS differences were
`1.41198/0.474854`, `3.84635/0.931005`, and `1.41198/0.474854`.

The missing contract is Storage lifetime, not Stream selection. Temporary model Tensors can be
destroyed before their non-default-Stream consumers complete. Per-destructor synchronization was
rejected because it would serialize the model.

The candidate API and tests are removed. Explicit `OpContext.stream`, caller-owned Graph replay
and `matmul_out_` remain. Future model propagation requires deferred release or an activation
arena before another ambient/explicit context experiment.

Full report: [Experiment 175](../optimization-log/experiments/175-scoped-model-stream-discard.md).
