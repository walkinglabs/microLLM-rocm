# Experiment 083 — 清空一个Cache row，而不碰其他请求

stop token已经告诉scheduler“这个请求结束了”，但static batch里的K/V Storage仍按batch row
排列。slot复用前必须证明：旧row可以在设备上清零，其他row保持不变，新写入不会重新暴露旧
prefix。

## API边界

```cpp
cache.clear_row(row);
```

它清空所有层、K和V、该batch row的**完整capacity**，不只清当前logical prefix。shared
`position()`保持不变；这是Storage所有权原语，不是假装已经有per-slot position。

未分配的Cache上调用是no-op；负数或超过batch的row稳定报错。FP32/BF16都使用逻辑0的相同
位表示。

## 测试场景

```text
2 layers · B2 · 1 KV head · capacity 6 · head dim 4 · BF16
```

清row 0共覆盖`2(K/V) × 2层 × 1 × 6 × 4 × 2 = 192 bytes`：

1. B2 prefix先写到position 3；
2. 保存row 1全部K/V；
3. `clear_row(0)`；
4. row 0全capacity为0、row 1 prefix逐项不变、position仍为3；
5. 共同decode写position 3后，row 0旧位置0–2仍为0，新位置有值；
6. CPU/HIP整个Cache逐项一致。

HIP clear在默认stream使用typed fill，测量区H2D/D2H calls均为0。原始GoogleTest JSON和环境
在[`083-data`](083-data/)。

最终门：CPU 201/201、HIP 88/88、ASan/UBSan 199/199、Torch-enabled 204/204；优化日志和
覆盖validator通过。

![KV Cache clear row](../assets/kv-cache-clear-row.svg)

## 决定与未完成项

保留。这个API证明旧数据可以安全删除和新位置可以再写，但全batch仍共享一个`position()`。
下一节点必须引入`positions[B]`及按row可见长度；在那之前，新请求不能在任意长度直接占用已清
slot，框架也不会声称支持真正continuous refill。
