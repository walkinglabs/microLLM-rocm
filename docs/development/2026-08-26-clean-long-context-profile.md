# 2026-08-26 — clean T2048/B2/N64基线与profile

三轮交替fresh process显示，microLLM/PyTorch为177.77/156.04 tok/s，即1.1393×；64个token全等，峰值
5.23/6.38GB，KV都为121,110,528 bytes且100%利用。旧0.8158×发生在materialized-score默认优化之前，
不再作为“当前”状态。

1/3-generation rocprof delta测得820.74ms Kernel：cached Attention finalize 42.27%，GEMM 33.25%，
scores 7.88%，cast 4.11%。allocator增量仍为0。下一节点必须审计一种没有被旧thread-map、split-PV、
exact-reuse或online实验覆盖的新finalize架构。
