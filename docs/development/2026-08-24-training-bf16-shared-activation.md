# 2026-08-24 — BF16训练共享激活原语

## 结果

仓库新增`bf16_gate_up_projection`和多输出Autograd接口，并让已有QKV投影进入相同图合同。
临时Transformer路由在三条策略上都未通过两模型性能门，因此源码中不保留模型或CLI开关。

## 设计边界

- forward只把共同FP32输入cast一次；
- 每个输出保留独立的图节点和weight父边；
- backward仍使用FP32 master input/weight；
- input gradient按普通Autograd分叉规则累加；
- Tensor shape、mirror dtype/device/contiguous条件由底层算子检查；
- CPU组合图、PyTorch oracle和HIP设备图覆盖全部输出与梯度。

## 为什么只保留原语

完整策略精确删除Qwen/DeepSeek的216/252次cast，但五进程Qwen只有`1.0066×`。QKV-only和
gate/up-only还分别出现`0.9804×`与`0.9911×`反例。原语可供未来图编译器或grouped GEMM组合，
当前eager Transformer继续走已验证路径。

完整记录见[实验212](../optimization-log/experiments/212-training-bf16-shared-activation-discard.md)。
