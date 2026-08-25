# 2026-08-25：Cached Attention三段检查器

## 为什么要再加一个接口

一次cached Attention可以先粗略理解为三步：

```text
query和历史key做乘法 -> score
score变成总和为1的概率 -> probability
概率对历史value加权求和 -> context
```

原来的融合算子一次做完三步。最终context错了时，我们不知道错误来自哪一步，也无法分别测量
三步的时间。已有`cached_gqa_attention_scores`打开了第一个盒子；这次增加诊断专用的
`cached_gqa_attention_context`，打开第三个盒子。中间仍复用公开softmax，因此现在每一步都能
独立观察。

这不是新的默认推理路线。诊断接口会把中间score和probability写入显存，实际模型仍使用融合
算子，避免为了方便检查而让正式推理变慢。

## 接口合同

- probability必须是连续FP32 `[batch, query_heads, 1, sequence]`；
- value cache必须是FP32或BF16 `[batch, kv_heads, sequence, head_width]`；
- `query_heads = kv_heads * repeats`；
- sequence区域必须连续，cache可以保留更大的head/batch stride；
- 输出固定为FP32 `[batch, query_heads, 1, head_width]`；
- CPU是容易阅读的参考循环，HIP复用当前context Kernel；
- shape、dtype、device或stride不满足合同时必须明确拒绝。

## 这次怎样证明它没算错

测试不是只看最后一个token。它保存并比较完整的score、probability和context Tensor：

- CPU手写小数值，并检查FP32/BF16与融合路径；
- PyTorch使用`softmax(Q @ K^T * scale) @ V`作为独立参考；
- MI300X使用DeepSeek形状H12/KV2/D128；
- batch为1和2；cache为FP32和BF16；
- sequence覆盖31/32/33、511/512/513和2048，共16个case；
- HIP计算区间不允许出现payload H2D/D2H，输入地址必须保持不变；
- 错误repeats和非支持stride必须被拒绝。

实际门禁结果：

| 门禁 | 结果 |
|---|---:|
| CPU Debug | 372/372 |
| ASan/UBSan CPU | 370/370 |
| 单卡MI300X HIP label | 191/191 |
| PyTorch operator parity | 1/1 |

HIP的16-case完整输出检查包含在191条单卡门禁中；三套CMake外部consumer也分别在CPU、
Sanitizer与HIP全套里通过。

## 仍然没有证明什么

接口正确不等于更快。下一节点才会用HIP Event和host wall time分别测Q·K、softmax、P·V与融合
baseline，并保留T2048反例。在三段完整精度门和至少1.05倍operator门通过之前，不改变正式模型
路由，也不宣称DeepSeek已经加速。
