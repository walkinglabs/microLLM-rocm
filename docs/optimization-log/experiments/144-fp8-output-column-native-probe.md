# Experiment 144：头文件说“可以”，运行时回答“不支持”

## 一个简单区别

“说明书里有一个按钮”和“这台机器按下按钮能工作”不是同一件事。hipBLASLt头文件定义了
权重侧outer-vector scale；我们必须真正提交一次，读取返回状态。

```text
先尝试 A-side outer-vector
  ├─ 成功：status=1，不需要post-scale
  └─ 被拒：status=0，缓存结果，改走scalar GEMM + device post-scale
```

![Output-column native capability](../assets/fp8-output-column-native-probe.svg)

## 结果

128² E4M3-FNUZ算子probe通过数值门，但记录：

```text
output_column_native_status = 0
output_column_scale_calls    = 1
software fallback calls      = 0
```

这表示矩阵乘法仍是原生FP8，只是列scale由随后一个GPU Kernel补回；不是整个算子退回软件GEMM。

两模型T512复核完全一致：

| 模型 | Linear | forward | post-scale | native status | fallback |
|---|---:|---:|---:|---:|---:|
| Qwen | 168 | 2 | 336 | 0 | 0 |
| DeepSeek | 197 | 2 | 394 | 0 | 0 |

hot-path column quantize为0，说明权重只在准备阶段处理一次。每个worker比较151,936个logits；
top一致但完整精度门仍失败，与Exp143结论相符。

## 决定

关闭“直接打开当前库的权重outer-vector即可恢复13%性能”这条解释。实现保留跨版本探测和安全
fallback，但本机后续优化不再重复试这个按钮。下一步只能减少需要per-column的Linear范围，或
把post-scale与相邻算子融合；先用权重误差分布选择范围，不能盲猜。
