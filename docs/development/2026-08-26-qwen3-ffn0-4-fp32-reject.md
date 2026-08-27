# Qwen3早期FFN FP32候选拒绝

日期：2026-08-26
状态：正确性门拒绝，未测性能

通用shape runner新增`--micro-bf16-ffn-fp32-layers`和scope元数据验证。候选完整运行
T1/32/128/512、B1/B2、prefill与N1/N4/N32 decode。

T128前缀8→22，但仍分叉；T512/B2/N32 batch invariant稳定3/3失败。常驻增量准确，不能抵消
正确性失败。结果保存完整64行raw、32-row summary与简明拒绝结论。默认策略不变。
