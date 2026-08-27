# Qwen3 phase策略扩展到T1024/T2048

日期：2026-08-27
状态：固定prompt显式证据扩展，限制保留

修复后的矩阵32/32 worker完成，16行中10 pass、4 precision mismatch、2 batch invariance
mismatch。12/12 cached行KV精确；microLLM六个B2 cached case全部保持相同行一致，Transformers
BF16在T1024/B2 N8/N32出现474/2行分叉。

T1024 B1/B2完整logit oracle都支持候选token2。T2048/B2支持候选token16而不是Transformers
BF16的220，但两个FP32实现Max为2.193e-4，略超2e-4门。因此合并结果是argmax 10/10、strict
common-FP32 8/10，不能写成完全向量对齐。

最大KV为477,102,080字节。T2048/B2/N32峰值microLLM/PyTorch为3.172/4.719GB；这是单进程
shape证据，不是重复性能排名。

显式策略的固定prompt边界扩到T2048，默认仍关闭。下一步需要不同prompt家族，而不是继续延长
同一个token重复序列。
