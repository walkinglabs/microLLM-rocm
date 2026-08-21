# Experiment 093 — position已经分叉，模型终于会算了

Experiment 084让KV Cache保存`row_positions[B]`，但模型仍只接受一个共同position。状态能表达
`[0,3]`，计算却会在`position()`处抛错；continuous slot refill因此还缺一条oracle路径。

## 第一版为什么故意串行

直接同时改RoPE、store和Attention三个HIP Kernel，出错时很难定位。第一版
`forward_cached_rows()`复用已验证B1路径：

```text
共享 [B,H,capacity,D] Storage
→ row 0 B1 view + position[0] → existing forward_cached
→ row 1 B1 view + position[1] → existing forward_cached
→ same-device合并 [B,1,V]
```

每个view共享原地址，只改变batch offset和logical prefix。因此每行自然得到自己的RoPE、K/V写入
位置和Attention可见长度。uniform positions直接走原并行batch快路径。

## 状态转移

| 事件 | row positions | 公共logical prefix |
|---|---|---:|
| B2 prefill | `[3,3]` | 3 |
| reset row 0 | `[0,3]` | 3 |
| divergent decode 1 | `[1,4]` | 4 |
| divergent decode 2 | `[2,5]` | 5 |
| reset最大row 1 | `[2,0]` | 2 |

最后一步同时修正`reset_row()`：当最高position被清空时，所有layer的logical Tensor view会缩到剩余
最大prefix，但底层capacity和Storage地址不变。

## 正确性证据

- B2 row 0逐项等于新建空Cache的独立B1；
- B2 row 1逐项等于保留3-token prefix的独立B1；
- 第二个decode step继续逐row对齐；
- FP32与BF16 Cache都通过；
- uniform API与原`forward_cached()`对齐；
- nonzero position却没有Storage会明确拒绝；
- HIP逐项接近CPU，执行区间D2H calls为0；
- key/value backing地址跨reset和两步decode不变。

![Divergent cached-row reference](../assets/divergent-cached-row-reference.svg)

## 明确没有完成什么

这条路径串行执行B个B1模型，并用same-device copy合并logits。它证明语义，不证明吞吐，也还没有
接进AdmissionBatchScheduler做真实slot refill。

下一节点可以安全拆成两个独立任务：先让scheduler调用这个oracle完成晚到请求正确性；再实现一次
并行的positions-aware RoPE/store/Attention Kernel，用本实验逐row结果作为不可降低的门。

数据见[`093-data`](093-data/)，入门图解见[不同页数的KV row](../../dev/divergent-kv-rows.zh-CN.md)。
