# Step 125 — Prefill Attention dual GEMM solution matrix

Status: planned

Experiment 308证明T2048有两个首差descriptor，不能只优化一个：

- QK在B4/B8先漂，B2 exact；
- P×V在B2先漂，而它的输入probability exact。

下一节点分别枚举当前gfx942/ROCm/hipBLASLt环境中的共同solution：

1. QK：`M2048 N2048 K128`，固定相同Q/K行，覆盖请求B1/B2/B4/B8对应batch count；
2. P×V：`M2048 N128 K2048`，固定相同probability/value行，至少覆盖B1/B2，并扩到B4/B8作边界。

每个候选依次通过：support → 完整CPU/默认reference → 跨batch row bitwise → 同batch重复行 bitwise →
HIP Event/wall。correctness失败者不计时。只把共同、零/可接受workspace、完整值通过的候选送入
DeepSeek模型反驳。QK与P×V允许需要不同solution；version-local index保持显式，默认不变。
