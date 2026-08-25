# 2026-08-25：Materialized-score多模型默认边界

## 矩阵目标

单个DeepSeek T2048通过，只能保留显式开关。自动默认必须回答：Qwen的D64 head是否也快？T512是否
能覆盖global score开销？B1/B2是否一致？

`materialized_attention_model_matrix.py`逐格调用已经验证的current/candidate成对runner：

```text
model   = Qwen2.5-0.5B / DeepSeek-Distill-1.5B
context = 512 / 2048
batch   = 1 / 2
```

每格保留独立子目录，里面仍有6个进程raw、3个pair、完整logits/token和图。外层只汇总，不丢弃子证据。

## 默认边界怎样决定

一个context只有在两个模型、两个batch全部同时满足下面条件时，才可成为minimum：

- 完整logits精度门通过；
- generated token完全相同；
- 三对吞吐中位至少1.05x；
- leave-one-pair-out最低至少1.01x；
- 所有更长的已测context也通过。

如果T512失败而T2048全过，建议minimum为2048；如果T2048仍有一格失败，返回0，表示不准入自动
策略。图中绿条表示精度和性能同时过门，红条保留反例。

伪runner合同覆盖2模型×2 context×2 batch共8格，故意让T512只有1.02x、T2048为1.20x，验证
minimum=2048、失败保留、子目录身份与SVG。下一提交运行真实官方权重矩阵。
