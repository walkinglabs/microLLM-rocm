# 2026-08-25 — gradient-as-bucket view结果

![Gradient-as-bucket views](../optimization-log/assets/data-parallel-gradient-bucket-views.svg)

Model-S三策略A/B中，bucket view相对persistent-copy再提升communication/total
1.131×/1.067×，相对transient为1.937×/1.367×。114个unpacked Storage和114次copy归零；
45个loss和9次参数门通过。

live回到transient基线，但peak仍多33,269,000B，所以不开默认。下一节点让Autograd直接向
bucket view累加，目标同时删除114次pack copy和backward双表示。

发布门：CPU `358/358`、ASan/UBSan `356/356`、RCCL标签`30/30`、118个测试文件注册。
