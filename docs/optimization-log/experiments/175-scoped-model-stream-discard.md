# Experiment 175 — Stream routing without lifetime routing corrupts the model

Status: `discard`; candidate fully removed

## Hypothesis

The existing model/autograd call tree mostly passes default `OpContext{}`. A thread-local,
lexically scoped Stream override could route that whole tree without editing every operator
signature. Explicit per-call Stream/external Stream would retain priority; nested scopes and
threads would be isolated.

This node intentionally did not change allocation or introduce a performance claim. The first
gate was exact model correctness on one tiny Transformer, followed by a capture-allocation
fallback probe.

## What passed before the model

- CPU nested scope restores outer/null state and a child thread sees no parent scope;
- caller-owned fill/add capture uses the ambient Stream and replays exactly;
- explicit per-call `OpContext.stream` overrides the ambient scope;
- wrong-device lookup fails;
- the installed-package consumer sees scope enter/exit correctly.

These local tests were insufficient because every buffer was explicitly owned until synchronization.

## Stable model failure

One-layer FP32 Transformer, vocabulary 16, width 16, B1 T4, fixed seed and 64 complete logits:

| Run | Max absolute difference | RMS difference | 1e-5 gate |
|---:|---:|---:|---|
| 1 | 1.41198 | 0.474854 | fail |
| 2 | 3.84635 | 0.931005 | fail |
| 3 | 1.41198 | 0.474854 | fail |

The output is not a small accumulation-order change. Nearly every logit changes materially.
After the deliberate capture-allocation failure, the next eager model call also reports
`embedding_kernel: operation failed due to a previous error during capture`.

## Cause

Changing submission Stream also changes lifetime requirements. On the legacy default Stream,
the current eager allocation/free behavior happens to preserve ordering assumptions. On a
non-default Stream, a C++ temporary can release its Storage while its queued Kernel output is
still consumed by later asynchronous work. The scope routed Kernels but did not route deallocation
or retain temporary owners.

Adding a synchronization to every destructor would make the result correct by serializing the
model and is explicitly rejected. Hiding the issue with a larger tolerance is also invalid.

![Scoped model Stream discarded](../assets/scoped-model-stream-discard.svg)

## Decision

Remove `ScopedOpStream`, ambient state, package exposure and all positive candidate tests. Retain
the failure data and an evidence validator that requires the unsafe API to stay absent.

The next prerequisite is lifetime-aware deferred release on an explicit Stream or a planned
activation arena. Only after that mechanism proves address/lifetime safety may model-wide Stream
propagation be retried.

Raw evidence is in
[`benchmarks/results/2026-08-24-scoped-model-stream-discard/`](../../../benchmarks/results/2026-08-24-scoped-model-stream-discard/).
