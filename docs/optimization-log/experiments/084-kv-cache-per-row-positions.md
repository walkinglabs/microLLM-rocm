# Experiment 084 — 一个batch里的row不必永远在同一页

Experiment 083能清空某一Cache row，但`position()`仍是全batch单值。新请求从0开始、旧请求
可能已到100；如果继续返回一个“看起来合理”的共同position，模型会静默读错K/V。

## 状态合同

KVCache现在保存`row_positions[B]`：

```text
row_position(i)   读取单row位置
advance_row(i,n)  只推进一row，带capacity检查
reset_row(i)      清完整row Storage并把该row位置归0
positions_uniform() 所有row是否相同
position()        仅uniform时返回；分叉时抛logic_error
```

旧`advance(n)`仍推进每一row各自位置；旧uniform训练/生成路径的行为不变。`reset()`清全部Storage
并把所有位置归0。

## 可执行状态转移

```text
[0,0,0]
→ advance_row(0,2) → [2,0,0]  position()拒绝
→ reset_row(0)     → [0,0,0]  uniform
→ advance(3)       → [3,3,3]
→ reset_row(1)     → [3,0,3]  position()拒绝
→ advance_row(1,3) → [3,3,3]  uniform
```

负row、越界row、非正count和超过capacity都稳定报错。HIP测试把真实B2 BF16 Cache从`[4,4]`
变成`[0,4]`，零payload transfer；补进4后重新uniform，CPU/HIP Storage仍一致。

![KV Cache per-row positions](../assets/kv-cache-per-row-positions.svg)

## 安全边界

现有`TransformerModel::forward_cached()`仍调用uniform `position()`，因此传入分叉Cache会明确
失败；它不会偷偷取max/min。下一节点才新增模型per-row cached API与按row可见长度Kernel。

原始CPU 2/2、HIP 1/1 GoogleTest JSON和环境在[`084-data`](084-data/)。本节点只完成状态模型，
不声称continuous batching已经可运行。

最终门：CPU 202/202、HIP 88/88、ASan/UBSan 200/200、Torch-enabled 205/205；优化日志和
覆盖validator通过。

## 决定

保留。显式失败比错误位置继续计算更重要。下一节点必须让RoPE、KV store和cached Attention
都消费`positions[B]`，并保持uniform旧路径数值不变。
