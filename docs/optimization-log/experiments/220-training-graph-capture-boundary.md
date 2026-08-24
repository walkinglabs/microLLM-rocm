# Experiment 220 — 能录下21个Kernel，不等于能重放一次训练

Status: `keep safety guard and probe; reject complete-training Graph`

## 问题

HIP Graph像把一串GPU工作录成一段视频。以后重放视频，可以减少CPU一次次提交Kernel的开销。
但视频中的每个Tensor地址必须一直有效，训练步数等状态也必须在重放时真的变化。

我们没有直接宣称“训练支持Graph”，而是把一个极小Transformer拆成四段：

```text
forward → backward → AdamW → full step
```

每段分别测试FP32和BF16，并在三个新进程中交替运行顺序。

## 第一个稳定失败：动态Storage会破坏capture

原始探针在forward/backward/full-step中调用同步`hipMalloc`。ROCm返回capture-invalidated，
而且当前gfx942运行时在`EndCapture`后仍把Stream报告为Invalidated；清理阶段再同步只会盖住第一
个错误。

修复不是吞掉错误，而是在普通Tensor Storage碰到HIP之前拒绝动态分配：

```text
开始capture
→ 标记当前线程正在capture
→ Tensor尝试申请新Storage
→ runtime立即给出“先预分配稳定workspace”
→ EndCapture正常结束
→ 同一Stream仍可执行eager fallback或下一次合法capture
```

新增真实MI300X测试证明：一次动态Tensor失败后，同一Scoped Stream可以立刻捕获一个预分配
`add_out`，重放结果为`[6, 8, 10, 12]`。

## 24进程分阶段结果

| Precision | Forward | Backward | AdamW | Full step |
|---|---|---|---|---|
| FP32 | 安全拒绝：动态Storage | 安全拒绝：动态Storage | 21 nodes captured | 安全拒绝：动态Storage |
| BF16 | 安全拒绝：动态Storage | 安全拒绝：动态Storage | 21 nodes captured | 安全拒绝：动态Storage |

![Training HIP Graph boundary](../assets/training-graph-capture-boundary.svg)

24/24进程都在失败后恢复为`capture status = None`，没有留下被污染的Stream。

## 第二个稳定失败：AdamW的GPU节点和CPU步数不是同一件事

AdamW阶段能够捕获21个设备节点。但是capture本身把主机`step`从0推进到1；随后重放Graph，
主机step仍是1。也就是说，GPU会再次使用第一次录制时的bias-correction常量，而训练器不知道
自己又走了一步。

因此下面这个推理是错误的：

```text
captured_nodes > 0
→ 完整optimizer可重放
```

正确结论是：设备工作可以录制，但变化的step、学习率和bias correction还没有设备所有权。

## 决定

- 保留capture期动态Tensor分配保护，防止ROCm失败路径污染Stream；
- 保留四阶段benchmark、三进程matrix和CPU schema测试；
- 不增加“完整训练Graph”开关，不报告吞吐加速；
- 下一版必须同时具备图级liveness plan、稳定forward/backward workspace和device-owned optimizer
  step，再进入端到端计时；
- 单独捕获AdamW节点不能作为训练优化合入。

原始证据位于
[`benchmarks/results/2026-08-24-training-graph-capture/`](../../../benchmarks/results/2026-08-24-training-graph-capture/)。
