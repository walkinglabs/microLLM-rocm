# Immediate default-stream exact-size pool

Experiment 087 removes the allocator's global 16-block retirement phase. Under the existing strict
legacy-default-Stream contract, a released exact-size address can be handed to the next Tensor
immediately because later GPU work is ordered after its previous use. Any non-default Stream still
permanently disables the pool.

No-sync stress tests, exact-size reuse tests and the non-default-Stream failure boundary pass.
DeepSeek T2048 B1/B8 alternating Release medians improve 1.010x/1.033x while backend allocations
fall to 94. Qwen/DeepSeek T512 B8 targeted rechecks improve 1.014x/1.099x. Peak, KV bytes and
baseline/candidate tokens remain unchanged.

The retained change stabilizes allocator behavior; it does not change Attention or model math.
See [Experiment 087](../optimization-log/experiments/087-immediate-default-stream-pool.md).
