# Model-S in-place bucket-average model gate

Experiment 256 compares one binary with only `--inplace-bucket-average`
changed. Both policies use Model-S B1T32, 25 MiB/3 buckets, five steps and a
final-step full parameter audit. Each policy has three alternating-order
processes; steady medians use steps 2–5.

| Policy | Average tensors | Backend allocs | Temporary bytes | Communication | Total | Peak |
|---|---:|---:|---:|---:|---:|---:|
| allocating | 6 | 126 | 374,068,224 | 6.60 ms | 19.21 ms | 603,383,808 B |
| in-place | 0 | 120 | 249,378,816 | 5.20 ms | 17.35 ms | 603,383,808 B |

Communication improves 1.269x and total improves 1.107x. All 30 loss values
match exactly, final rank parameters match, and peak memory is unchanged.

In-place averaging remains the default. It stabilizes bucket Storage addresses
and removes one full rank-local bucket representation, but 120 backend
allocations and 228 pack/unpack copies remain. Persistent bucket plus unpacked
gradient storage is the next reducer milestone.

