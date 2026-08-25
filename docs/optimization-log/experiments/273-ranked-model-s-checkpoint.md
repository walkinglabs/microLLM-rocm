# Experiment 273 — 178.4 MiB完整状态能否由两个rank恢复

Status: `Model-S checkpoint smoke complete`

固定Model-S、T32、两rank。中断路径1步checkpoint+恢复1步，对照不中断2步。

| Evidence | Result |
|---|---:|
| checkpoint size | 187,042,096 bytes |
| rank0 write | 1,022–1,068 ms |
| rank1 wait max | 1,069 ms |
| read/verify max | 532 ms |
| load+restore max | 740 ms |
| resumed/uninterrupted checkpoint | byte-identical |
| 57 tensors / 15,586,176 values | rank Max/RMS 0 |

![Ranked Model-S checkpoint](../assets/ranked-model-s-checkpoint.svg)

完整checkpoint包含62.34MB模型和两组AdamW FP32 moments，加上格式字段后约178.4MiB。三个成功
组都只有rank0写，rank1等待marker后读取；final step与optimizer step均2。

失败传播继续复用tiny注入，因为它测试同一barrier/write/marker层且避免故意制造187MB失败文件。
rank0=1、peer=−15。checkpoint、safetensors、ready、tmp和ID全部删除。

这些单次I/O数字只证明当前环境资源可用，不是磁盘性能排名。Model-S checkpoint smoke完成；下一
分布式结构缺口转向world-size 4接口/失败边界。

证据：[`ranked Model-S checkpoint`](../../../benchmarks/results/2026-08-25-ranked-model-s-checkpoint/)
