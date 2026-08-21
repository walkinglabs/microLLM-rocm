# Shared-cache row prefill reference

`TransformerModel::forward_prefill_cached_row()` now admits one non-empty `1xT` prompt into one
empty row of a shared batch `KVCache`. It computes the prompt through the existing B1 full-prefill
path, copies every active K/V head into the target row on the same device, advances only that row,
and returns `[1,1,V]` last-token logits.

The contract rejects non-empty or out-of-range rows and incompatible Cache metadata. CPU and HIP
tests cover FP32/BF16, stable shared Storage, exact preservation of the other row, independent-B1
prefill and continued-decode equality, and zero HIP execution D2H payload copies.

This is deliberately a temporary-B1 correctness reference. It is not a throughput claim and does
not yet make the scheduler continuous. See [Experiment 094](../optimization-log/experiments/094-slot-row-prefill.md)
and the [beginner guide](../dev/slot-row-prefill.zh-CN.md).
