# 同一个 prompt 换一行：怎样检查 B2 prefill 有没有写错位置

Experiment 104已经知道B1和B2 prefill会产生不同logits，但还有两种解释：

1. BF16矩阵在不同batch shape下有小数值差异；
2. B2的第二行、stride或KV复制写错了位置。

第二种是实现错误，必须修；第一种则要用误差和外部参考判断，不能要求所有shape逐位相同。

## 四个最小用例

目标是原来分叉的32-token prompt，记作P5。另一个同长度prompt记作P4：

| case | 输入 | P5所在local row | 用途 |
|---|---|---:|---|
| single_5 | `[P5]` | row0，B1 | 单行参考 |
| pair_4_5 | `[P4,P5]` | row1，B2 | 原顺序 |
| pair_5_4 | `[P5,P4]` | row0，B2 | 交换顺序 |
| duplicate_5 | `[P5,P5]` | row0和row1，B2 | 相同行检查 |

每个case运行三个fresh process。`--continuous-prompt-offsets`让runner显式指定P4/P5，而不是靠
请求编号暗中生成不同输入。

## 看见了什么

P5的第一个prefill选择：

| 路径 | row | top-1/logit | top-2/logit | margin |
|---|---:|---|---|---:|
| B1 single | 0 | 151643 / 12.352085 | 151648 / 10.704218 | 1.647867 |
| B2 `[P4,P5]` | 1 | 151643 / 12.297267 | 151648 / 10.771706 | 1.525561 |
| B2 `[P5,P4]` | 0 | 151643 / 12.297267 | 151648 / 10.771706 | 1.525561 |
| B2 duplicate row0 | 0 | 151643 / 12.297267 | 151648 / 10.771706 | 1.525561 |
| B2 duplicate row1 | 1 | 151643 / 12.297267 | 151648 / 10.771706 | 1.525561 |

四条B2记录逐值相同。完整16-token输出也完全相同，并且都在生成位置4选择1196；B1选择23606。

## 结论怎样成立

如果row1 stride错，P5从row1换到row0后，错误应跟着row1走；实际没有。如果KV copy把两行混在
一起，两行相同prompt或交换顺序后应出现不同；实际B2两行从prefill logits到完整输出都一致。

因此当前证据推翻了“local row/stride/KV copy错误”。B1/B2差异只跟计算shape变化。结合默认B2
在原请求上匹配PyTorch，最强解释变成BF16/hipBLASLt在M=32与M=64下的数值路径差异。

这仍不是“所有算子已经证明正确”。下一步要保存B1/B2完整logits的max-abs/mean-abs，并在每个
Transformer block找出差异首次明显增长的位置。

![B2 prefill row audit](../optimization-log/assets/prefill-row-audit.svg)

完整证据见[Experiment 105](../optimization-log/experiments/105-b2-prefill-row-audit.md)。
