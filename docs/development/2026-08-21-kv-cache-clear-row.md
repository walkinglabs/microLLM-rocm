# KV Cache row clearing

`KVCache::clear_row()` zeros one batch row across every layer's full backing capacity while keeping
the current shared logical position. It constructs a contiguous full-capacity view of the row and
uses the existing typed fill path, so HIP performs no payload transfer.

CPU/HIP tests cover BF16 B2 storage, invalid rows, undefined-cache no-op, preservation of the other
row, and a subsequent shared decode. Old prefix positions remain zero while the new position is
writable. This is a storage ownership primitive; per-slot positions remain intentionally separate.

See [Experiment 083](../optimization-log/experiments/083-kv-cache-clear-row.md).
