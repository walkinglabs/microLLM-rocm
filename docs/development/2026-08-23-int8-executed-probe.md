# 2026-08-23：raw INT8 executed probe

## 实现

独立benchmark直接调用hipBLASLt INT8×INT8→INT32，不修改公共Tensor枚举。输出完整矩阵后，
每个shape固定5个位置由CPU整数点积验证exact。

## 结果

- 6/6 size、30个抽样点exact；
- 128→4096为0.40→416.03 TOPS；
- 4096官方peak/roofline利用率15.91%；
- pre/post设备门通过。

## 边界

不存在公共INT8 Tensor、量化器、scale/zero-point、Transformer Linear或模型推理。下一步若继续，
先设计weight-only per-channel scale合同，不能把raw TOPS写成模型支持。

详见[Experiment 121](../optimization-log/experiments/121-int8-executed-probe.md)。
