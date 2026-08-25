# Experiment 232 — 关闭一条路线后，重新看时间花在哪里

Status: `profile accepted; select T1024 Attention GEMM solutions`

## 为什么不能沿用旧热点

online模型路线经历了private kernel、public operator、三cast模型门和direct-BF16反驳，最终关闭。
继续凭旧T512 trace选任务会把已经变化的shape和策略混在一起。因此重新固定当前默认路径：

- 两个官方revision；
- B1T1024；
- BF16 FFN/Attention、FFN/QKV Arena、exact grouped QKV/gate-up；
- BTHD与retained BF16 Q/K开启；
- online Attention明确关闭且调用计数为0。

每个模型跑`load+1`和`load+6`，用`(6−1)/5`得到steady prefill。rocprof原始trace留在临时目录，
仓库保存四份Kernel stats、四份CLI route记录和两个delta。

## 新时间线

| Category | Qwen | DeepSeek |
|---|---:|---:|
| hipBLASLt GEMM | 59.7% | 66.8% |
| causal softmax | 14.8% | 9.2% |
| other kernels | 10.3% | 10.3% |
| FP32/BF16 cast | 5.8% | 4.3% |
| RMSNorm | 3.5% | 3.4% |
| GQA repeat | 3.1% | 3.3% |
| add | 2.7% | 2.7% |

![Current inference profile](../assets/current-inference-profile.svg)

Qwen/DeepSeek总Kernel时间为8.315/14.862ms。softmax是最大单个具名kernel，但Experiment 207已在
T1024测得128-thread只快1.013×/1.021×，折算整步不到0.3%，局部线程track保持关闭。online
Attention也已经用完整模型反例关闭。

## 决定

- 不重新调softmax threads，不重开online模型；
- 下一节点只筛T1024 QK/PV四个exact hipBLASLt问题；
- 必须先完整输出再计时，三进程取共同candidate；
- 即使算子winner成立，仍要过双模型完整logits/peak/端到端门；
- profile节点不改变任何默认代码。

原始证据位于
[`benchmarks/results/2026-08-25-current-inference-profile/`](../../../benchmarks/results/2026-08-25-current-inference-profile/)。

本节点不改核心源码；上一节点完整CPU/HIP 537/537、RCCL 14/14、multi-GPU 12/12继续适用。
新增profile schema在CPU、ASan、PyTorch-enabled与HIP四个配置中定向通过，累计CTest计数为
342/340/316/538，覆盖清单注册104个测试文件。
