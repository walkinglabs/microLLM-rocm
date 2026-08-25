# Model-S direct bucket-gradient gate

Experiment 259 compares transient, bucket-view, and direct-accumulation policies in
one binary. Every policy uses Model-S B1T32, 25 MiB/3 buckets, five steps, and a
final parameter audit. Three process runs rotate policy order; steady medians use
steps 2–5.

| Policy | Forward/backward | Communication | Total | Peak bytes |
|---|---:|---:|---:|---:|
| transient | 12.105 ms | 6.740 ms | 19.630 ms | 603,383,808 |
| bucket views | 10.400 ms | 3.585 ms | 14.900 ms | 636,652,808 |
| direct | 12.535 ms | 1.650 ms | 15.035 ms | 623,447,040 |

Direct accumulation removes all 114 pack and unpack copies and makes communication
2.173x faster than bucket views. It also saves 13,205,768 peak bytes. However,
forward/backward falls to 0.830x because each operator still creates its gradient
output and leaf accumulation adds it into the prepared view. Total reaches only
0.991x of bucket views, so the model route is rejected.

All 45 losses and nine final parameter audits match exactly. The independently
tested leaf accumulation-target primitive remains as a foundation for producer
out-kernels; the C++/CLI direct reducer route is removed in the next code node.
