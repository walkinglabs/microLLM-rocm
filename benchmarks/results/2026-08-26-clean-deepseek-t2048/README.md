# Clean DeepSeek T2048/B2/N64 baseline

Three alternating fresh-process pairs compare the current clean microLLM Release
against PyTorch ROCm. All 64 tokens match. Median throughput is 177.77 versus
156.04 tok/s, or 1.1393x. Peak memory is 5.23 versus 6.38 GB; both KV caches are
121,110,528 bytes at 100% utilization.

This replaces the old current-baseline claim. It does not rewrite the historical
0.8158x result that preceded the retained materialized-score optimization.
