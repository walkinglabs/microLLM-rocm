# 2026-08-26 — 删除失败的FFN gate/up模型route

Experiment 319已经拒绝选择性与all-batch策略，Experiment 320也完成了最后一次因果trace。本提交删除：

- `PrefillFfnGateUpProjection`枚举值与model context；
- `--fp32-prefill-ffn-gate-up-solution-index`及注册、输出逻辑；
- 选择性模型、all-exact模型和post-exact gate/up三个candidate runner；
- 只验证这些已删除runner的测试。

保留通用FP32 row-invariance工具、FFN filtered trace基础设施、所有raw结果、SVG与实验文档。结果测试改为
只读固定证据，并新增源码/文件缺失合同，防止失败用户路径悄悄回来。
