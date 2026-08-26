# FP32 Attention request-batch solution matrix

This result screens the real DeepSeek T2048 strided-batched descriptors:

- QK: `M2048 N2048 K128`, transpose K;
- P×V: `M2048 N128 K2048`;
- request batches B1/B2/B4/B8 become backend batch counts 12/24/48/96.

Every common solution passes support, complete finite output, CPU sentinel, and every
repeated request block before it is timed. The formal run uses one process, one warm-up,
and three measured Event/wall samples. It is an operator screening gate, not an
end-to-end performance claim.

Results:

- QK: 34 common, 34 exact across batch, zero non-regressing candidates.
- P×V: 2 common, 2 exact across batch, zero non-regressing candidates.
- Both defaults drift across request batch on identical inputs.
- `admitted_index` is `-1` for both operations.

The best exact QK/P×V indices, 304681 and 295716, are retained only as a model-level
counterfactual. Their minimum operator speedups are 0.9159× and 0.5354×.

Reproduce:

```bash
python3 benchmarks/single_gpu/fp32_attention_batch_invariance_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_fp32_attention_batch_invariance \
  --output-directory /tmp/fp32-attention-batch-invariance \
  --warmup 1 --repetitions 3
```

See [`summary.json`](summary.json), [`analysis.json`](analysis.json), and
[`verification.json`](verification.json).

![solution matrix](attention-solutions.svg)
