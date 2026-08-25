# Experiment 281：当前0.8158x差距到底花在哪里

Status: measured baseline; admit cached-Attention microarchitecture track

## 当前跨框架事实

DeepSeek T2048/B2/N64三对fresh process中，microLLM/PyTorch为133.50/163.64 tok/s，即
0.8158x。64个token完全一致；峰值为5.23/6.38GB；两边KV都是121,110,528 bytes且100%
利用。旧0.868x不再代表当前环境。

## profile方法

同一个干净二进制分别运行`load + warmup + 1 generation`和`load + warmup + 3 generation`。
Kernel统计做`(three - one) / 2`，得到一次T2048/B2/N64 generation。rocprof同时保存HIP API、
copy和allocation统计。

## 结果

![Current DeepSeek T2048 profile](../assets/current-deepseek-t2048-profile.svg)

| 类别 | 每generation时间 | Kernel占比 | 调用 |
|---|---:|---:|---:|
| cached Attention | 647.3 ms | 61.57% | 1,792 |
| hipBLASLt GEMM | 270.4 ms | 25.72% | 12,861 |
| 其他Kernel | 52.7 ms | 5.02% | 7,720 |
| cast | 32.2 ms | 3.07% | 7,308 |
| RMSNorm | 21.0 ms | 2.00% | 3,705 |
| KV store | 6.8 ms | 0.65% | 1,792 |

1,792恰好是28层×64 token。单次fused cached Attention约361.2us。

每generation有224次/117.44MB D2D、1次16KB H2D、1次512B D2H。backend allocation增量为0，
36,963次逻辑申请全部cache reuse；旧allocator相位问题不是当前第一热点。

HIP API delta中`hipMemcpy`显示327.6ms，但同步API会等待之前GPU工作，不能把它写成纯copy成本。
Kernel分类和应用wall互相支持cached Attention第一，而不是复制或KV store第一。

证据：[`current DeepSeek T2048 profile`](../../../benchmarks/results/2026-08-25-current-deepseek-t2048-profile/)

## 决定

Step 105准入cached-Attention score/context微架构矩阵。先在DeepSeek H12/KV2/D128、T512/T2048、
B1/B2上分别测score和context，守住完整score→probability→context门；在算子矩阵过门前，不连接
模型、不重开allocator、不优化只占0.65%的KV store。
