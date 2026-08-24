# 2026-08-24 — Hybrid BF16 AdamW

## 改动

- `AdamW` 支持只为不超过阈值的 BF16 moment Tensor 建立 multi-tensor workspace；
- 大 Tensor 继续走独立向量 Kernel；
- Auto 在 HIP BF16 moment 模式下解析为 1,048,576 elements；
- `0` 禁用，正数用于研究覆盖；
- CLI/JSON 报告解析后的阈值、Tensor 数、元素数和 metadata 传输；
- CPU/HIP 测试覆盖小/大 Tensor 混合、完整状态和默认 Auto 解析。

## 证据

六个阈值、72 个 pilot 进程，加 20 个正式进程。1M 正式 Qwen/DeepSeek optimizer 为
`1.2404×/1.2631×`，端到端为 `1.0490×/1.0528×`。16M 的 DeepSeek 反例为
`0.8956×/0.9800×`。

完整说明见 [Experiment 215](../optimization-log/experiments/215-hybrid-bf16-adamw.md)。

最终发布门：CPU 324/324、ASan/UBSan 322/322、PyTorch 298/298、CPU/HIP
508/508（3 个条件跳过、HIP 标签 173/173）、RCCL 14/14（multi-GPU 11/11）。干净覆盖率为
79.8% lines、87.7% functions、60.4% branches。
