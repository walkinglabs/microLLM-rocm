# Qwen3对称投影校准

日期：2026-08-26
状态：选择down-FP32进入下一门

gate/up/down FP32三规则使用相同五case。gate为4/5；up/down为5/5。up/down常驻和BF16
tensor数相同，down-FP32最小margin更大，因此选择它进入完整shape，不提前合入。

结果保存两份完整summary和40行raw；Python evidence固定选择规则、margin、常驻与pending状态。
