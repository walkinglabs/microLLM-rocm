# FP8 global scale grid

新增独立runner，在运行前固定4个activation scale和4个weight scale。每个官方模型先生成一份
FP32完整logits，再用fresh process执行16个FP8候选；选择规则先看完整logits门，再看top token、
RMS、max error，最后才看速度。

CPU标签回归225/225通过，2个条件跳过；runner合同6/6通过。MI300X正式34/34 worker成功，
但32个FP8候选0个过门。Qwen最佳点落在activation上边界0.05，因此下一步扩展边界，不能提前
把有限网格写成普遍结论。

详见[Experiment 123](../optimization-log/experiments/123-fp8-global-scale-grid.md)。
