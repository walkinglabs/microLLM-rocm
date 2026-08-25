# Model-S persistent gradient-bucket gate

Experiment 257 compares one binary with only `--persistent-gradient-buckets`
changed. Both policies use Model-S B1T32, 25 MiB/3 buckets, in-place averaging,
five steps and a final-step parameter audit. Each policy has three alternating-order
processes; steady medians use steps 2–5.

| Policy | Later backend allocs | Communication | Total | Live bytes | Peak bytes |
|---|---:|---:|---:|---:|---:|
| transient | 120 | 7.070 ms | 21.025 ms | 498,757,632 | 603,383,808 |
| persistent | 0 | 4.205 ms | 16.360 ms | 623,447,040 | 761,342,216 |

Communication improves 1.681x and total improves 1.285x. All 30 loss values
match exactly, all six final rank-parameter audits report zero difference, and
all 12 later persistent steps allocate zero communication Storage.

The speed result is real, but the first implementation keeps both six bucket and
114 unpacked-gradient Storage objects. Live bytes rise by 124,689,408 and peak
bytes by 157,958,408, so the route remains explicit rather than becoming the
default. The next experiment replaces unpacked Storage and its 114 copies with
contiguous parameter-shaped views into the reduced buckets.
