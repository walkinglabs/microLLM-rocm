# Qwen3 phase policy across four prompt-content patterns

The previous long-context matrix repeated token 1. This node changes only prompt content while
keeping the Qwen3 revision, weights, phase policy, contexts, batches and measurement method fixed.

Patterns are generated from a local base manifest with:

```bash
python3 benchmarks/single_gpu/make_qwen3_prompt_pattern_manifest.py \
  --input /path/to/qwen3-runtime-manifest.json \
  --output /tmp/qwen3-prompt-pattern-manifest.json
```

The matrix covers four seeds, T512/T2048, B1/B2, prefill and cached N8: 64 workers and 32 aggregate
rows.

## Result

- 64/64 workers complete;
- 29 direct pass and three visible `precision_mismatch` rows;
- all three mismatches belong to the constant seed and match previously audited FP32 argmax states;
- alternating, ascending and historical-sensitive seeds pass 24/24 aggregate rows directly;
- both frameworks keep identical B2 rows in all eight cached B2 cases;
- all 16 cached rows have exact and cross-framework-equal KV bytes;
- maximum KV is 471,597,056 bytes;
- maximum microLLM/PyTorch engine peaks are 3,166,208,000 / 4,713,611,776 bytes.

The phase policy introduces no new split in the three changed-content seeds. This supports a narrow
content-robustness claim for fixed token patterns, not a general natural-language prompt claim.
Every shape has one process per framework, so throughput is not ranked here.

Files:

- `prompt-seeds.json`: portable seed definitions;
- `matrix-*`: 64 worker records and 32 aggregate rows;
- `oracles/`: the three constant-seed first-split oracle summaries;
- `summary.json`: compact decision.
