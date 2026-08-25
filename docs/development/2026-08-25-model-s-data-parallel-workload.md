# 2026-08-25 — Model-S 成为真实多 bucket 双卡 workload

distributed CLI 新增 `--model tiny|model-s`、`--context` 和 `--batch`。每步记录model shape、
参数量、bucket参数/元素数量和两卡最大engine peak；执行参数审计的step若rank差非零会直接失败。

真实Model-S结果：

- 参数：15,586,176；
- B1T4单步：12 buckets、57个参数Tensor、rank差0、单卡peak约549 MB；
- B1T32三步：step 2的forward/backward 13.77ms、communication 16.33ms；
- step 3末步审计约246.73ms，参数差仍为0。

这终于提供自然多bucket workload。下一步扫描1/4/25MiB，而不是在人为tiny bucket上实现overlap。

