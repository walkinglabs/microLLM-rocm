# 2026-08-24 — Packed weight-gradient 失败

新增包含实际D2D pack成本的独立benchmark和三进程runner。四个官方case数值正确，但速度
`0.835×–0.979×`，因此没有修改Autograd或模型。

详细证据见
[Experiment 218](../optimization-log/experiments/218-packed-weight-gradient-discard.md)。

发布门：CPU 327/327、ASan/UBSan 325/325、PyTorch 301/301、CPU/HIP 511/511
（3 个条件跳过、HIP 标签 173/173）、RCCL 14/14；覆盖清单注册 90 个测试文件。
