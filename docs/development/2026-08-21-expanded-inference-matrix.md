# Expanded inference context, batch and memory matrix

## Why

A single short prompt and batch one cannot answer whether serving scales. The existing official
Qwen/DeepSeek runner already separated prefill, cached decode and uncached decode, but readers had
to derive batch efficiency and whole-device memory share themselves.

## Added contracts

- named `smoke`, `standard`, and `extended` suites include short, medium, long and boundary contexts;
- the standard batch axis is B1/B2/B4/B8; extended adds B16;
- every successful row reports the physical device capacity and peak share;
- summaries report peak and KV bytes per request, throughput per peak GiB, B1-relative throughput
  scaling, batch efficiency and peak-memory scaling;
- cached rows reject an active Cache larger than their allocated Storage;
- the summary freezes the exact context, batch and case axes used by the run.

The fast executable gates cover 18 CPU and 24 HIP combinations. They use three context lengths,
B1/B2/B4 on CPU, B1/B2/B4/B8 on HIP, and both FP32/BF16 Cache. Every HIP output row must equal the
CPU result, while the CPU test independently checks the exact allocated Cache formula.

The measurements continue to use fresh processes and alternating microLLM/PyTorch order. A shape
that is unsupported or out of memory remains a result rather than disappearing from the report.

## Measured result

The completed MI300X survey contains 120/120 successful process rows. BF16 Cache is exactly half
the FP32 Cache at all 12 representative shapes and preserves microLLM's token suffix. The wider
matrix also rejects a short-prompt generalization: Qwen T2048 B8 prefill is 0.173x PyTorch and
DeepSeek is 0.465x. DeepSeek T2048 cached decode remains below PyTorch, while Qwen remains above it.

Cross-framework generation is not universally exact: Qwen passes 18/18 cached comparisons;
DeepSeek passes 10/18, with retained first-difference indices. The two frameworks still use
different resident precision policies, so this is a recorded alignment failure and a request for
a matched-dtype full-logit experiment, not proof of which implementation is wrong.

See [Experiment 076](../optimization-log/experiments/076-expanded-inference-service-matrix.md).
