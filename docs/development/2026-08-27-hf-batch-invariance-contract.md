# 相同输入batch不变量成为一等证据

日期：2026-08-27
状态：合同与官方smoke通过

旧PyTorch worker在相同batch行生成不同token时直接抛异常，只留下`failed`。现在保存全部
`generated_rows`、跨测量确定性和`generated_rows_equal`，aggregate先分类
`batch_invariance_mismatch`，再判断首行跨框架token。

Qwen3 T1024/B2/N8中，microLLM两行都在敏感step选2；Transformers BF16 row0选474，row1选2。
独立B1完整logit oracle确认两个FP32实现和phase候选都选2。两框架KV均为236,716,032字节。

合同测试83/83通过。该smoke只有每框架一个进程，不作性能排名；修复后的完整长上下文矩阵必须
重新运行，旧generic failure不能混入新结果。
