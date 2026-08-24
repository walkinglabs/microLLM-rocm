# FP32 Attention 精确算法注册与整模验收

日期：2026-08-24

## 用一句话说明

我们给 hipBLASLt 的算法编号加上了一把“只匹配这一台环境和这一种矩阵”的锁，
然后在真实 Qwen 和 DeepSeek 上检查：编号确实能运行、结果能完全相同，但整台模型
没有稳定快过 1%，所以没有把它偷偷设成默认值。

## 为什么不能直接写死一个编号

同一个“矩阵乘法”在底层还包含很多信息：有多少个 head、矩阵怎样转置、每一批隔多远、
允许多少临时显存、GPU 是哪种架构、HIP 和 hipBLASLt 是什么版本。算法编号只在这些条件
完全相同时才有意义。少比较一个条件，就可能把能跑 A 的编号错误地交给 B。

新 `Fp32MatmulSolutionKey` 因此保存：

- batch、输入和输出的实际行列数；
- 三个 batch stride 和两个 transpose 标记；
- GEMM alpha 的精确位模式、执行模式和 workspace 上限；
- GPU architecture、HIP runtime/driver 和 hipBLASLt 版本。

第一次使用时还会问当前 hipBLASLt：“这个编号真的支持这份 descriptor 吗？”答案为否就
直接报错，不会悄悄换另一个算法。之后按 device index 分开缓存解析结果，统计 registry
hit/miss、cache hit/miss 和真实 dispatch 次数。

## 先发现了一次精度反例

算子实验最快的 QK 编号只有约 `1e-7` 误差，看起来很小。但模型有 24 或 28 层，误差会经过
softmax、残差和下一层继续传播。pilot 的最终 logits Max 变成 `0.07290` 和 `0.04437`。

所以我们没有放宽门槛，而是回到候选表，改用算子输出 bit-exact、同时仍比默认快的 QK 编号。
替换后，24 个正式进程的完整 logits Max/RMS 都是 0。

## 正式测量

两模型各测 baseline、只换 QK、只换 PV、两者都换。每种策略 3 个独立进程，每个进程
2 次热身、5 次测量；第 2 个进程把顺序反过来。

- Qwen：QK `1.0093×`，PV `1.0037×`，both `1.0082×`；
- DeepSeek：QK `0.9991×`，PV `1.0029×`，both `1.0043×`；
- 六行显存峰值、engine allocation 次数都与 baseline 相同；
- 候选都记录了正确数量的注册项、缓存和 dispatch。

## 结论

基础设施保留，因为它让以后任何算法编号都必须经过精确条件和整模证据。默认优化拒绝，
因为没有一种策略在两模型上同时达到 `1.01×`。下一次 Attention 计算优化必须减少更大区域
的工作，例如把 GEMM 周围的 scale、softmax 或 layout 一起融合；继续换编号已经不是新假设。
