# Experiment 371 — 全模型gate保留FP32，能成为简单校准规则吗

Status: `4/5 oracle cases; calibration rule rejected`

![Qwen3 gate-FP32 rejection](../assets/qwen3-bf16-gate-fp32-reject.svg)

候选不是再拼层：所有FFN gate FP32，up/down与Attention BF16，BF16 Cache。五个首次分叉状态
使用共同FP32 oracle预筛。

T32/B1、T32/B2、T128/B2和强制T512/B2匹配FP32；T512/B1失败。该格FP32选2955，候选与
Transformers BF16都选1096，候选错误margin 0.003286。4/5不满足全部case门。

候选在32-row和性能前拒绝。这个结果只关闭gate-FP32规则；对称up/down-FP32必须使用同一
五case门另测，不能从一个投影失败推广到另外两个。
