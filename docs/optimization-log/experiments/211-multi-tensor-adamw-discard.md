# Experiment 211 — 一个Kernel更新许多参数

Status: `discard model route; keep primitive`

## 先看见浪费

Qwen有290个参数Tensor，DeepSeek有339个。旧AdamW每个Tensor启动一次Kernel。一次“热身+两步
测量”因此出现870次和1,017次AdamW启动。计算公式完全相同，只是参数放在许多不同内存块里。

新原语先做一张不会变化的“街区地图”：每个GPU block应该更新哪个Tensor。每一步只上传一张
很小的“地址卡”，里面是参数、梯度、两个moment和可选BF16 mirror的地址。Kernel根据地图找到
地址卡，再更新对应元素。

```text
stable block map ─┐
                  ├─ one multi-tensor AdamW Kernel
step descriptors ─┘       ├─ FP32 parameter
                           ├─ first / second moment
                           └─ optional BF16 mirror
```

## 第一次失败：异步名字不等于异步行为

第一版用同步H2D上传地址卡。Qwen/DeepSeek端到端只有`1.0533×/0.9927×`。虽然Kernel少了，
同步copy却先等待尚未完成的backward，破坏了队列顺序。

第二版给地址卡使用pinned host staging，在同一HIP Stream异步copy。一个Event只保护“下次CPU
改写地址卡”不早于上次copy完成；它不增加全局同步。运行时仍准确记录每步一次metadata H2D。

## 正确性门

- 三个不同大小Tensor，覆盖256尾部、多个block和缺失gradient；
- 完整比较parameter、first moment、second moment和BF16 mirror；
- Scalar与multi最大差异不超过`2e-6`，不谎称bit-exact；
- 缺失gradient的完整状态保持不变；
- 测量区间每步正好一次metadata H2D，payload D2H/D2D都是0；
- CPU拒绝HIP workspace，空workspace stats可查询；
- native default-Stream async copy有CPU/HIP生命周期与counter测试。

## 五进程正式门

两个模型均为BF16 Linear + FP32 master、B1/T512、一次热身、两步测量；策略顺序交替，每个策略
五个新进程。

| Model | Per Tensor | Multi Tensor | Speedup | Peak ratio | Loss relative diff | Parameter diff |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 14,742.46 | 15,587.30 tok/s | 1.0573× | 1.00065× | 0.133% | 0 |
| DeepSeek Distill 1.5B | 6,226.66 | 6,285.24 tok/s | 1.0094× | 1.00075× | 0.0043% | 0 |

DeepSeek比`1.01×`少约0.0006，门不能事后降低。

## Profile解释

| Model | Old AdamW calls/time | Multi calls/time | Kernel speedup | All calls saved |
|---|---:|---:|---:|---:|
| Qwen | 870 / 16.667 ms | 3 / 11.339 ms | 1.4699× | 864 |
| DeepSeek | 1,017 / 48.450 ms | 3 / 44.743 ms | 1.0828× | 1,014 |

Qwen主要受launch影响，所以收益明显。DeepSeek的大Tensor让每个旧Kernel已经足够长；合并启动
不能改变AdamW必须读写parameter、gradient和两个moment的带宽成本。它连隔离的`1.10×`门也
没有通过。

![Multi-tensor AdamW discard](../assets/multi-tensor-adamw-discard.svg)

## 决定

删除Transformer训练器和CLI的临时接入，普通AdamW行为不变。保留公开multi-tensor primitive、
pinned staging、native-stream async copy、完整状态测试和profile。下一版若继续，必须减少block
map读取、共享descriptor或使用更合适的分块，而不是再次只减少launch。

原始证据在
[`benchmarks/results/2026-08-24-multi-tensor-adamw/`](../../../benchmarks/results/2026-08-24-multi-tensor-adamw/)。
