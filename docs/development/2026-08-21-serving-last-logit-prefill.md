# Serving last-logit prefill

## Problem

The official inference matrix called a full `[B,T,V]` output a serving prefill. Profiling showed
that its largest single kernel projected every historical hidden state into the vocabulary, then
the benchmark copied almost 10 GB of logits to the host. A generator needs only the final position.

## Interface

- `forward_inference()` remains the explicit full-logits reference and returns `[B,T,V]`.
- `forward_inference_last_logits()` processes the same context but projects only the final hidden
  position and returns `[B,1,V]`.
- `microllm_hf_infer --prefill-logits last|full` makes the benchmark contract visible; `last` is
  the serving default.
- the PyTorch oracle uses `logits_to_keep=1` for the same last-logit contract.
- uncached generation also uses the last-logit API instead of forming and slicing full logits.

## Correctness

CPU MHA/GQA tests gather the final rows from full inference and compare every value. HIP tests
cover B2, output shape, no hidden payload transfer and a CPU oracle. Official Qwen/DeepSeek T2048
full-vs-last comparisons keep the same top token with max absolute error below `3.1e-5`.

## Measured impact

At T2048 B8, three-process medians improve microLLM by 2.97x for Qwen and 1.32x for DeepSeek.
Peak engine memory falls by 74.0% and 65.0%; measured D2H shrinks exactly 2048x. Profiled output
head GEMM time falls by more than 99%, while causal softmax stays constant and becomes a clean next
hotspot.

See [Experiment 077](../optimization-log/experiments/077-serving-last-logit-prefill.md).
