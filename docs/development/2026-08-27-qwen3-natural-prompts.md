# Qwen3四条exact自然prompt

日期：2026-08-27
状态：四prompt显式策略通过，两个batch-dependent边界保留

固定tokenizer生成英文22、中文15、代码18和chat-template24 token；每条使用原始长度，不重复填充。
32/32 worker完成，16行中14 pass、2 precision mismatch、0 batch失败，KV8/8精确。

英文B1 step6候选4416、Transformers785；B2双方4416。中文B1双方104136；B2候选104136、
Transformers3837。四份B1/B2完整logit oracle全部strict通过，并支持候选。代码与chat 8/8直接一致。

双方四个B2 case都保持行一致；这里是跨batch算法选择变化，不是同一B2内部行分叉。最大KV7.34MB，
峰值microLLM/PyTorch为1.866/1.315GB；单进程不作性能排名。

显式策略证据扩到四条短prompt，默认仍关闭。下一步应扩大真实prompt数量和长度，或回到训练/硬件
尺度；四条样例不能证明语言质量。
