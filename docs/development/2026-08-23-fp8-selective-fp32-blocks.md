# FP8 selective FP32 block counterfactual

`ModelConfig::fp8_fp32_layers`是严格递增、范围内的block索引列表，仅在FP8 Linear策略下允许。
模型构造时，选中block的7个Linear改为FP32；Norm本来就是FP32，其他block与output head保持FP8。

准备阶段只收集真正的FP8 Linear。两层untied tiny模型选择block1后，转换block0七个Linear和
output head一个，共8个；block1七个权重保持FP32。CPU lazy/prepared输出一致。

HIP门证明mixed prepared模型执行、0 H2D/0 D2H，动态量化调用只来自未选block和output head。
完整Release/MI300回归353/353通过，2个条件跳过，fresh CLI binary contract包含
`--fp8-fp32-layers`。

该接口用于可反驳的精度实验，不是自动精度策略。正式反事实固定Qwen21、Deep27；应同时报告
converted tensors、resident memory、TPS和完整logits，不能只看关键层trace。

正式Exp140已完成：两模型各18个worker全部正常结束，每个worker比较151,936个logits。
T8 RMS分别改善3.21%/13.41%，但T512 RMS分别恶化3.89%/6.05%，同时常驻权重增加
42.66/133.87 MiB，T512吞吐下降1.47%/3.49%。所以接口保留为诊断工具，但关键层策略不默认
开启。完整证据见[Experiment 140](../optimization-log/experiments/140-fp8-selective-block-counterfactual.md)。
