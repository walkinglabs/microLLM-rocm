# Qwen3 down-FP32候选拒绝

日期：2026-08-26
状态：扩展oracle拒绝

down-FP32通过原5case和正式短性能，但完整矩阵新增T128/B1/B2分叉。新增B1 full-logit oracle
明确候选25、FP32/Transformers320。保存64行shape、6行性能和两份新oracle。

down-FP32不合入；up-FP32因新增状态也匹配FP32，进入下一完整shape门。
