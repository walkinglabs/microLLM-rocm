# Experiment 283：一个head只有一个Block，确实浪费了MI300X

Status: operator candidate admitted; model route pending

## 搜索为什么从120条扩到144条

初始矩阵搜索S1/2/4/8/16，8个winner全部落在S16边界。边界winner不能称为最优，所以增加允许
上限S32，重新运行完整144个新进程，而不是只给一个S32幸运点。

固定DeepSeek H12/KV2/D128、T512/T2048、B1/B2、FP32/BF16 cache。48个candidate各有3个
进程，每进程3次热身和20次Event/wall测量，current/split顺序交替。

![Split-sequence search](../../../benchmarks/results/2026-08-25-cached-attention-split-matrix/split-search.svg)

## 八个winner

| T | B | cache | 最佳S | current ms | split ms | Event加速 | wall加速 | partial bytes |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 512 | 1 | BF16 | 16 | 0.0730 | 0.0305 | 2.389x | 2.084x | 99,840 |
| 512 | 1 | FP32 | 16 | 0.0970 | 0.0302 | 3.211x | 2.743x | 99,840 |
| 512 | 2 | BF16 | 16 | 0.0730 | 0.0307 | 2.381x | 2.084x | 199,680 |
| 512 | 2 | FP32 | 16 | 0.0963 | 0.0303 | 3.178x | 2.710x | 199,680 |
| 2048 | 1 | BF16 | 32 | 0.2736 | 0.0397 | 6.897x | 5.856x | 199,680 |
| 2048 | 1 | FP32 | 32 | 0.3712 | 0.0458 | 8.096x | 6.988x | 199,680 |
| 2048 | 2 | BF16 | 32 | 0.2727 | 0.0495 | 5.511x | 4.868x | 399,360 |
| 2048 | 2 | FP32 | 16 | 0.4186 | 0.0671 | 6.238x | 5.666x | 199,680 |

winner Event为2.381x–8.096x，wall为2.084x–6.988x。所有完整context Max/RMS最多
3.90e-9/1.09e-9；计时区间0 payload transfer、0 warm backend allocation。最大partial只有
399,360 bytes。

## 反例证明因果关系

S1在八格全部失败，最低只有0.546x。它保留第二次launch和partial traffic，却没有增加blocks。
S2在八格全部过1.05，最低1.185x。这个对照支持“新增sequence并行度带来收益”，而不是误把
两阶段公式或测量噪声写成原因。

S32也不是普遍最好：T512四格都由S16胜出，T2048/B2/FP32同样回落到S16。策略必须看shape，
不能硬编码“split越多越好”。

## 决定

准入显式官方模型A/B，不改默认。下一步给uniform cached decode增加固定S开关；先比完整logits、
64个token、allocation/peak，再跑DeepSeek T2048/B2/N64三对fresh process。只有端到端收益通过，
才讨论自动policy。

证据：[`cached Attention split matrix`](../../../benchmarks/results/2026-08-25-cached-attention-split-matrix/)
