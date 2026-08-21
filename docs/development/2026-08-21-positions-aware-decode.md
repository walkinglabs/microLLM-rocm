# Positions-aware parallel cached decode

Active continuous rows now execute one batched Transformer path. Device Int32 `positions[A]` and
`cache_rows[A]` drive per-row interleaved/split-half/bias RoPE, mapped FP32/BF16 K/V pair stores,
and cached GQA Attention with independent visible prefixes. Prefixes above 4096 use a masked
scores/softmax/context fallback.

CPU/HIP operator tests compare every positioned primitive with scalar row references. Model tests
compare complete active-row logits with independent B1 models, preserve inactive full capacity,
cover Q/K bias and require zero execution D2H.

Three alternating Release A/B matrices retain median speedups of 1.295x (R8/S2), 1.670x (R8/S4)
and 1.610x (R4/S4); all 18 processes preserve one checksum per shape. A single R8/S2 matrix
regression was retained and then rebutted by all three alternating pairs.

See [Experiment 098](../optimization-log/experiments/098-positions-aware-decode.md) and the
[beginner guide](../dev/positions-aware-decode.zh-CN.md).
