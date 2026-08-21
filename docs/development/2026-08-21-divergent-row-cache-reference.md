# Divergent cached-row reference execution

`TransformerModel::forward_cached_rows()` now consumes unequal `KVCache::row_positions()`. The
correctness-first implementation creates B1 views over shared batch Storage, reuses existing
cached forward semantics, merges row logits, and advances each row independently. Uniform rows
retain the original batch fast path.

CPU and HIP tests cover FP32/BF16, two decode steps, independent-B1 equality, zero execution D2H,
stable backing addresses, missing-storage errors and logical-prefix shrink after resetting the
maximum row. The path is deliberately serial and is the oracle for future positions-aware kernels.

See [Experiment 093](../optimization-log/experiments/093-divergent-row-cache-reference.md) and the
[beginner guide](../dev/divergent-kv-rows.zh-CN.md).
