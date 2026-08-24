# Experiment 221 — 把计分牌放进GPU，Graph才知道自己走了几步

Status: `keep explicit primitive and FP32 many-small candidate; reject universal/BF16 routing`

## 从Experiment 220的失败开始

上一轮AdamW能捕获设备节点，但Graph重放后主机step仍停在1。原因很简单：录像里只有GPU
Kernel，记录步数的计分牌却在CPU手里。重放录像不会自动改CPU变量。

本次增加一个稳定的device-owned状态：

```text
device int32 step
device float corrections[2]
          │
          ├─ advance Kernel：step += 1，计算两个bias correction
          └─ 所有AdamW更新Kernel从同一地址读取correction
```

这两个Tensor在capture前分配，地址在所有重放期间不变。checkpoint或回到普通`step()`前，
调用者必须显式把device step同步回optimizer；未同步时读取state、加载state或走普通step都会被
拒绝，避免保存一个过期步数。

## 正确性门

- FP32 moment与BF16 moment各捕获一次、连续重放三次；
- 参数、一阶moment、二阶moment和BF16 mirror全部与普通eager AdamW对齐；
- Graph后同步step为3，再走一次普通step，两条路径仍在step 4对齐；
- 从step 4重新建立device state并捕获一次，恢复起点到step 5仍对齐；
- timed region没有H2D、D2H或D2D payload transfer；
- PyTorch AdamW oracle扩到第三步，标准host path继续直接对齐；
- 正式矩阵每个进程执行3次热身+50次测量，最终device step都是53。

正式60进程中，完整状态sample的最大误差为`7.45e-8`；BF16 moment sample最大误差更小。

## 为什么性能不是“Graph一定快”

| Moment | Tensor形状 | Graph/eager wall |
|---|---:|---:|
| FP32 | 1 × 1K | 0.230× |
| FP32 | 16 × 1K | 0.991× |
| FP32 | 64 × 1K | 1.427× |
| FP32 | 256 × 1K | 1.436× |
| FP32 | 16 × 256K | 0.869× |
| BF16 | 1 × 1K | 0.193× |
| BF16 | 16 × 1K | 0.677× |
| BF16 | 64 × 1K | 0.772× |
| BF16 | 256 × 1K | 0.805× |
| BF16 | 16 × 256K | 0.817× |

![Device-owned AdamW Graph step](../assets/adamw-graph-replay.svg)

FP32的小Tensor很多时，CPU提交64/256个Kernel的成本明显，Graph减少提交后获益。单Tensor、
大Tensor和BF16 moment则被Graph节点调度、device correction读取和更短Kernel的固定成本吃掉。
尤其BF16 Kernel原本工作更少，固定开销占比更大。

## 决定

- 保留`AdamWGraphStepState`和FP32/BF16 graph-replayable更新原语；
- 保留显式`synchronize_graph_step`，checkpoint不会静默读取旧step；
- 不把Graph设为AdamW默认，不对BF16 moment启用；
- FP32、每Tensor约1K元素、至少64个Tensor是下一轮multi-tensor/整模候选；
- 下一反驳应把稳定descriptor和device step合成两节点multi-tensor Graph，检查能否消除BF16的
  每Tensor节点成本；如果大Tensor仍回退，就关闭optimizer-only Graph track。

原始数据位于
[`benchmarks/results/2026-08-24-adamw-graph-replay/`](../../../benchmarks/results/2026-08-24-adamw-graph-replay/)。
