# Experiment 290：线程少一半，为什么几乎没有更快

Status: performance rejected; explicit research seam retained

## 假设

finalize原来每个block有256个线程，但Qwen head width只有64，DeepSeek只有128。看起来一半甚至
四分之三线程在P×V阶段没有负责column。候选用64/128个物理线程模拟完全相同的256个逻辑lane，
保留每条局部累加流、共享归约树和P×V position顺序。

![Finalize mapping matrix](../../../benchmarks/results/2026-08-25-cached-attention-finalize-mapping/mapping.svg)

## 完整矩阵

两模型、T512/T2048、B1/B2、FP32/BF16、64/128/256线程，每个策略两个fresh process，共96条。
16/16 case的64/128输出都与current逐项位级相同，热backend allocation为0。

| 映射 | Event speedup范围 | wall speedup范围 | 过门case |
|---|---:|---:|---:|
| 64 vs 256 | 0.5548x–0.9651x | 0.5691x–0.9704x | 0/16 |
| 128 vs 256 | 0.9901x–1.0121x | 0.9808x–1.0121x | 0/16 |

目标DeepSeek T2048/B2/BF16是最好的一格，也只有Event/wall 1.0121x/1.0121x，远低于
1.05x/1.02x双门。64-thread在D128上还要让每个线程模拟四个逻辑归约lane并负责两个column，
Event最低只有0.5548x。

## 推翻了什么

“空闲线程很多”没有推出“减少线程就能减少finalize时间”。主要成本仍是每个column按T逐项读取
value并串行累加；物理线程减少没有缩短这条链，也没有增加独立block数量。

## 决定

不进入官方模型A/B，不修改Auto。显式线程参数和runner作为可复现研究接口保留。Step 107完成，
Step 108只移动P×V归约：score和softmax继续走exact-order路径，P×V按sequence拆分后合并。若模型
logits仍出现不可接受漂移，就能把问题定位到value累加顺序，而不是softmax。

证据：[`finalize mapping matrix`](../../../benchmarks/results/2026-08-25-cached-attention-finalize-mapping/)
