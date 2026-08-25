# Experiment 225 — 什么时候可以安全地把default-Stream内存池重新打开

Status: `keep explicit runtime primitive; continue three model cases`

## 从上一个拒绝继续

非默认Graph Stream一出现，旧runtime永久关闭exact-size pool。这保证安全，却让原本稳定的Qwen
gradient地址全部失配。

本次不是直接把`enabled=true`，而是新增一次可证明的阶段交接：

```text
quiesce_and_enable_hip_caching_allocator(device)
→ hipDeviceSynchronize：等待所有Stream完成
→ 清除“非默认工作未证明结束”的禁止状态
→ 开始新的default-Stream-only复用阶段
```

任何后续非默认提交都会在enqueue前再次调用`notify_non_default_stream`关闭pool。覆盖入口包括
Graph capture/launch、Event record/wait、显式OpContext、async copy和stream-ordered allocation。

## runtime生命周期门

- 非默认Stream创建后，普通`enable_hip_caching_allocator`仍然拒绝；
- quiescent handoff后，两个8KiB default-Stream Storage为1次backend allocation+1次reuse；
- 同一条旧Stream记录Event会再次关闭pool；
- 再次quiesce才能恢复，计数从1变2；
- Graph capture后quiesce可恢复，但Graph launch会再次关闭；
- 这些检查在同一GTest进程顺序运行也通过，不依赖每个CTest新进程清空状态。

## 24进程模型preflight

| Model | Context | Handoff off | Handoff on | Handoffs/run | Graph launches |
|---|---:|---:|---:|---:|---:|
| Qwen 0.5B config | 8 | rejected | rescued | 3 | 0 |
| Qwen 0.5B config | 512 | rejected | rescued | 3 | 0 |
| DeepSeek-Distill 1.5B config | 8 | rejected | rescued | 3 | 0 |
| DeepSeek-Distill 1.5B config | 512 | rejected | still rejected | 3 | 0 |

![Quiescent allocator handoff](../assets/quiescent-allocator-handoff.svg)

每个策略/case三个新进程。关闭策略12/12失配；handoff让pool保持enabled并救回三个case。
DeepSeek T512仍失败，与Experiment 223纯default-Stream的198项变化反例一致，说明handoff不会
伪造地址稳定。

## 决定

- 保留显式quiescent API和重新提交即关闭的状态机；
- 默认行为不变：没有调用新API时仍是保守disable；
- Qwen T8/T512、DeepSeek T8进入下一轮真实Graph optimizer性能/数值门；
- DeepSeek T512继续禁止，不降低门；
- 下一实验的每个训练step必须按`handoff→backward→snapshot check→Graph launch→sync`执行；
- 如果每步device-wide sync吃掉optimizer收益，就拒绝模型路由并转向Event粒度阶段交接。

原始证据位于
[`benchmarks/results/2026-08-24-quiescent-allocator-handoff/`](../../../benchmarks/results/2026-08-24-quiescent-allocator-handoff/)。
