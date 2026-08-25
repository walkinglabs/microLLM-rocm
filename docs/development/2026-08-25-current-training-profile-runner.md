# 2026-08-25 — 训练 profile 也要能一条命令重跑

旧的训练热点报告已经能解释 `load + 1 step` 与 `load + 3 steps` 怎样相减，但采集过程仍要
手工拼四条命令。手工作业容易漏掉一个开关，最后比较的就不是同一条训练路径。

新增 `profile_current_training.py`，它把当前保留合同固定为：

- 两个 pinned 官方模型；
- batch 1、context 512；
- BF16 Linear 和 FP32 master weight；
- BF16 AdamW moments 与 1,048,576-element hybrid threshold；
- tied-embedding sparse add 开启；
- 已保留的 Attention layout fusion 开启；
- 已拒绝的局部策略全部显式关闭；
- optimizer measured window 内 Tensor payload 的 H2D/D2H 必须为零；保留的 descriptor
  metadata 每步精确为 Qwen 13,888 bytes、DeepSeek 12,608 bytes，且 D2H 为零。

runner 为每个模型启动 `load + 1 step` 和 `load + 3 steps` 两个新进程，保存应用 JSON、
Kernel CSV 和派生 profile。静态合同测试已注册到 CTest；真实性能仍必须在 MI300X 上执行，
不能用合同测试代替。
