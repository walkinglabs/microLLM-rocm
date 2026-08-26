# Experiment 324 — 原生128归约也只有约1.003×

## 结果

16个进程覆盖T512/T2048、B1/B2、FP32/BF16 cache。完整输出全部通过，Max≤3.73e-9；native128与当前
materialized256不位级相同，说明新tree确实执行了。

性能没有兑现：T2048四个case的Event和wall都只在约1.003×，0个达到1.05×/1.02×门。候选拒绝，
不进入完整模型。

这也推翻了“P×V有128线程闲置，所以改成128线程会明显加速”的主要解释。瓶颈更可能受顺序P×V内存/
指令流限制，而不是物理线程数本身。candidate代码删除后，finalize局部搜索停止。

![Native128 result](../../../benchmarks/results/2026-08-26-native128-finalize/native128.svg)
