# 2026-08-21 — inference context, batch and KV-cache matrix

## Delivered

- official Qwen/DeepSeek paired inference runner;
- context, batch, prefill, cached and uncached decode axes;
- fresh-process order alternation and median summaries;
- explicit pass/unsupported/OOM/failed rows;
- Storage allocated bytes, active bytes, capacity, utilization and peak share;
- separate microLLM/PyTorch precision residency policies;
- cache-prepare, steady-decode and end-to-end timing fields;
- prefill top-token/logit and decode-token alignment fields;
- five Python contract tests and a CTest registration;
- a beginner-oriented design guide and repository-owned SVG.

## Evidence outcome

The 108-row core matrix passes and all decode token pairs match. The 48-row batch matrix
has 42 pass rows and six intentional microLLM `unsupported` cached-batch rows. The 24-row
warm long-context matrix passes and reveals severe prefill/cached-decode gaps. A separate
36-row no-warm-up matrix is retained only for feasibility and invalidation evidence.

The old fixed short-prompt 4/4 performance result does not generalize. Experiment 060 is
therefore a kept measurement/infrastructure node and a falsification result, not a speedup.

See [the report](../optimization-log/experiments/060-inference-context-batch-kv-matrix.md),
[raw evidence](../optimization-log/experiments/060-data/) and the
[simple design guide](../dev/inference-matrix.zh-CN.md).
