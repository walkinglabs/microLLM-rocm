# Step 168 — official Qwen3 complete one-step parameter audit

Status: complete

## 问题

gate/up的56个Tensor通过FP32门，不能证明Attention、Norm、Embedding、down和输出别名。

## 计划

1. 对齐C++内部名字和Hugging Face名字；
2. 在AdamW前导出全部梯度，一步后导出全部参数；
3. 检查311个存储Tensor到310个独立Tensor的tied alias；
4. 用固定Max/aggregate-RMS门分别跑FP32和BF16；
5. 保存逐Tensor、族别、worker证据，删除临时大payload；
6. 画一张能同时显示通过门与失败家族的SVG。

## 结果

FP32比较1,192,099,840个值并通过全部固定聚合门。BF16完整运行但Gradient Max和
Parameter RMS失败；最坏梯度从gate/up扩展后落在tied embedding。

详细证据见[Experiment 383](../experiments/383-qwen3-training-all-parameter-audit.md)。
