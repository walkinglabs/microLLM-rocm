# 2026-08-24 — FP32 weight-gradient solution 反例

扩展FP32 exact solution registry到显式rank-2 key，并新增官方gate/up weight-gradient tuner。
三进程共同index在算子层提高Qwen/DeepSeek `1.077×/1.133×`；模型层精确命中144/168次，
但端到端为`0.993×/0.996×`，因此不设默认。

详细证据见
[Experiment 219](../optimization-log/experiments/219-fp32-weight-gradient-solutions-discard.md)。

发布门：CPU 329/329、ASan/UBSan 327/327、PyTorch 303/303、CPU/HIP 514/514
（3 个条件跳过、HIP 标签 174/174）、RCCL 14/14；覆盖清单注册 92 个测试文件，
覆盖率为79.8% lines、87.7% functions、60.4% branches。
