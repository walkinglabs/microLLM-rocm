# 2026-08-24 — Grouped weight-gradient 能力失败

新增独立 FP32 weight-gradient GroupedGemm benchmark 和 8-case runner。Qwen/DeepSeek 的
QKV/gate-up在direct `N,T`与materialized `N,N`两种布局下均为0 supported candidate。

模型和Autograd没有改动。测试覆盖runner的8格完整性与discard schema。详细记录见
[Experiment 217](../optimization-log/experiments/217-grouped-weight-gradient-discard.md)。

发布门：CPU 326/326、ASan/UBSan 324/324、PyTorch 300/300、CPU/HIP 510/510
（3 个条件跳过、HIP 标签 173/173）、RCCL 14/14；覆盖清单注册 89 个测试文件。
