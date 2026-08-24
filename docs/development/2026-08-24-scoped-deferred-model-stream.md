# Lifetime-safe model Stream routing

## Problem

The removed Stream-only scope changed tiny Transformer logits by as much as 3.846 because queued
non-default-Stream consumers outlived temporary Tensor Storage. The explicit lifetime primitive
then proved safe release, but callers still had to route every operator manually.

## Implementation

- added `runtime::ScopedDeferredHipStream` with inseparable work/lifetime routing;
- resolved otherwise-default `OpContext::native_stream` calls through the active scope while
  retaining the existing inline device-selection boundary;
- routed runtime strided-copy Kernels through the same seam;
- rejected nesting, different Streams and different devices;
- retained thread isolation, capacity/byte counters and explicit `finish()`;
- added CPU rejection, HIP operator/layout, tiny full logits and complete forward/backward tests;
- added an official-model benchmark and alternating-process matrix runner.

## Evidence and decision

All tiny and official correctness gates pass exactly. The 48-process official matrix rejects
default enablement: inference reaches 0.125×–0.862× and training 0.235×–0.575× of legacy, while
one DeepSeek T512 training step retains 15,591,456,776 deferred bytes.

Profiler work count is identical. The current exact-size pool is deliberately legal only on the
legacy default Stream, so candidate model calls replace cache reuse with thousands of synchronous
backend allocations/frees. The API remains explicit infrastructure; production model paths stay
unchanged.

Full analysis: [Experiment 177](../optimization-log/experiments/177-scoped-deferred-model-stream.md).

## Regression note

CPU 281/281, ASan/UBSan 279/279, PyTorch-enabled CPU 255/255 and CPU+HIP 434/434 pass.
The current RCCL model set passes 6/11: five rank-local model tests fail while pure collectives
pass. A detached fresh build of pre-node `adcd642` reproduces the identical hipBLASLt invalid-
device failure, proving it is not introduced here. It is retained as the next dedicated
multi-GPU correctness node rather than hidden behind this single-GPU experiment.
