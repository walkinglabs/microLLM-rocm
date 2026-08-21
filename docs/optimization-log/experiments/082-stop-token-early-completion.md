# Experiment 082 — 模型说“结束”时，请求应当立刻进入终态

此前请求只能达到`max_new_tokens`或被外部取消。真实模型会生成EOS/stop token；如果scheduler
不知道这个信号，就会继续浪费decode step，也无法判断slot何时可补新请求。

## 公共合同

`GenerationConfig::stop_tokens`是一组唯一、合法的词表ID。生成token先被追加到输出；如果它
属于stop集合，本请求随即结束，不再调用下一次decode。空集合保持旧行为。

Scheduler快照新增独立原因：

```text
None | Length | StopToken | Cancelled
```

`completed_requests`仍统计所有正常终止，`stop_completed_requests`单独统计内容早停。

## 单请求与batch语义

单请求B1 Cache在命中stop的同一步释放，已生成前缀保留。static batch中，各row可以不同时间
结束；已结束row不再追加输出，也不再推进随机数，但为了维持当前统一cache position，会向该
row喂一个被忽略的合法dummy token，直到其他row结束。

这意味着：

- 逻辑输出和独立生成一致；
- 已结束row不会污染其他row；
- 但static batch的物理slot/Cache row尚不能中途回收。

最后一点不是隐藏失败，而是下一continuous batching节点的明确输入条件。

## 证据

- CPU 4/4：单请求立即早停、不同row长度、B1 Cache释放、Admission completion reason；
- HIP 2/2：不同长度row逐项对齐CPU、scheduler当步释放Cache；
- duplicate/out-of-range stop token稳定报错；
- stop token按集合语义排序后加入Admission compatibility key：顺序不同的同一集合仍能合组，
  内容不同的终止合同不会误合组。

原始GoogleTest JSON与环境在[`082-data`](082-data/)。本节点不声称吞吐提升；它建立slot
refill需要的真实终止信号。

最终门：CPU 200/200、HIP 87/87、ASan/UBSan 198/198、Torch-enabled 203/203；全部目标
warning-clean构建，优化日志与覆盖validator通过。

![Stop-token early completion](../assets/stop-token-early-completion.svg)

## 决定

保留。下一节点必须把“逻辑结束”升级为“物理slot可复用”：需要per-slot position、Cache row
reset/replace和不会让新请求看见旧K/V的测试。在这些完成前，不能称当前static batch为
continuous batching。
