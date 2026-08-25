# 2026-08-25 — ranked checkpoint ownership infrastructure

one-process-per-GPU训练已有rank identity、persistent views、context-selective overlap和peer失败传播，
但checkpoint仍缺“谁写、何时写、其他rank怎样知道写完”的合同。

## Worker contract

- `--resume-file`在任何bucket plan/hook建立前恢复模型、AdamW moments/step和ExperimentState；
- 每次optimizer step同步推进`global_step`与确定性data cursor；
- checkpoint前用一个rank all-reduce barrier证明所有rank完成optimizer；
- 只有rank0调用现有完整checkpoint API；它先原子写`.tmp→checkpoint`，再原子发布ready marker；
- 非0 rank不写文件，只等待匹配`step=N`的marker，然后读取checkpoint验证step/cursor；
- 输出明确记录initial/final/optimizer step、resume、requested/written/verified；
- rank0写失败时不发布marker，launcher终止等待peer。

## End-to-end launcher

`run_ranked_checkpoint.py`执行：

```text
2-step rank group -> rank0 interrupted checkpoint
new rank group restores -> 3 more steps -> resumed-final checkpoint
fresh uninterrupted rank group -> 5 steps -> uninterrupted-final checkpoint
```

最终不仅比较所有参数，还要求两个完整checkpoint逐字节相等；这同时覆盖模型、AdamW两组moments、
optimizer step和ExperimentState。成功后删除checkpoint、ready、tmp和communicator ID，不提交二进制。

## Pilot

- resumed/uninterrupted final checkpoint均10,796 bytes且逐字节相等；
- 两rank与恢复/不中断参数最大差均0；
- rank0写3次，非0 rank写0次；final experiment/optimizer step均5；
- rank0写失败明确返回1，等待peer返回−15，未留下任何文件；
- 初始5秒故障门被约5.2秒进程启动抢先触发，两个进程都被终止；修正为15秒后根因门通过。

完整RCCL标签49/49，checkpoint静态合同通过，测试文件125。pilot只准入正式tiny证据，Model-S
仍需单独处理大checkpoint与完整rank参数比较。
