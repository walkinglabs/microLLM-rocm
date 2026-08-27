# Qwen3 global up-FP32 candidate rejection

The candidate keeps every FFN up projection in FP32 while converting FFN gate/down and all
Attention projections to BF16.  It is evaluated as a global policy: the same representation and
dispatch rule is used during prefill and cached decode.

## Result

- complete shape matrix: 64/64 fresh workers and 32/32 aggregate rows complete;
- KV evidence: all 24 cached rows have exact capacity, active bytes and theoretical bytes;
- answer evidence: 23 rows match Transformers BF16 directly; all nine mismatch rows are attributed
  at their first shared-input split to a common PyTorch FP32 oracle;
- extended oracle: up-FP32 matches FP32 in all eight unique states, including new T128/B2 step22
  and T512/B2 step2 states;
- performance: four cached-decode cases pass the 0.95 throughput and 1.05 latency gates, but
  T512/B2 prefill reaches only `0.8875x` throughput and `1.1268x` latency;
- five-case throughput geometric mean: `0.9578x`, below the fixed `0.97x` gate;
- memory: resident weights increase exactly 176,160,768 bytes; incremental model peak stays within
  the declared tolerance in every case.

The global candidate is rejected.  Precision improvement does not override a repeatable prefill
regression.  The projection-scoped implementation remains an explicit diagnostic capability; it is
not selected as the default policy.

## Reproduce

The exact 64-worker shape command is saved in `shape-command.txt`.  The performance gate is:

```bash
HIP_VISIBLE_DEVICES=1 python3 \
  benchmarks/single_gpu/compare_qwen3_up_fp32_matrix.py \
  --manifest /tmp/microllm-qwen3-runtime-manifest-v2.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory /tmp/qwen3-up-fp32-performance-matrix
```

Every performance case uses three fresh processes per policy, alternating policy order, with two
warm-up iterations and five measured iterations.  Cross-policy token changes are not judged by the
performance script: each policy must be deterministic, while the complete-logit oracle files decide
whether a changed token is correct.

## Files

- `shape-summary.json` / `shape-raw.jsonl`: complete 32-row / 64-worker matrix;
- `t128-b2-step22-oracle-*`: new long-decode first-divergence audit;
- `t512-b2-step2-oracle-*`: new batch-two first-divergence audit;
- `performance-summary.json` / `performance-raw.jsonl`: five cases and 30 fresh processes;
- `summary.json`: final decision and compact evidence index.
