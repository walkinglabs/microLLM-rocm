# 2026-08-25 — ranked world-size infrastructure

`RankCommunicator`支持一般world size，但worker/launcher此前写死2。本节点泛化同一正确性合同。

## Generalized path

- worker接受正world size，CPU reference拼接`world_size×context`global batch；
- rank-local synthetic batch对rank≥2也确定且不同，rank0/1历史输入保持不变；
- checkpoint ExperimentState记录真实world size；
- worker JSON回报world size；
- launcher按`N−1…1,0`启动，rank0最后发布ID；
- 进程输出按真实rank重排，参数/CPU/loss/timing/plan/overlap门覆盖全部rank；
- Model-S临时safetensors按rank命名并逐一对照rank0与CPU；
- 通用matrix的rank process计数使用`policy runs × world size`；
- 故障注入使用非法rank N，并终止其余peer。

world-size 1 tiny一步通过，CPU最大参数差`1.4e-8`；world-size 2原有普通/bucket/failure三条真实
CTest全部通过。

## Four-rank boundary

当前4个MI300X VF上的tiny一步在约2.7秒内失败。四个rank均返回1，stderr相同：

```text
ncclCommInitRank: unhandled system error
```

`/dev/shm`容量为67,108,864 bytes（64 MiB），与历史四卡RCCL共享内存边界一致。新增
`failure-mode=group-init`把这种环境能力失败记录为结构化通过：world size、四个return code、
system-error rank数、组耗时和共享内存容量；若未来初始化成功，则继续完整训练/reference门。

完整RCCL标签49/49，测试文件125。下一提交从干净revision保存world1/world2成功与world4稳定
失败证据；不得声称当前环境4卡训练成功。
