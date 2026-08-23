# Experiment 133 data

同一个multi-block Tensor amax改动的两套独立反驳实验：

- `weight/`：Qwen/Deep T8 device weight amax冷启动，18条；
- `activation/`：Qwen/Deep T512 host-weight + device activation amax，18条；
- `fresh-*`：独立Release Ninja HIP gfx942+hipBLASLt 34-step构建。

两套都使用FP32完整词表logits、BF16对照和三次fresh-process轮换。候选与各自单block基线的
max/RMS逐值相同，因此性能差异可归因于reduction并行度；精度门仍失败。
