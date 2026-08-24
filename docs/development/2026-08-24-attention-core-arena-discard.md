# Attention core Arena experiment

## 实现

- `CausalGqaAttentionWorkspace`保存scaled Q、expanded K/V复用槽和probabilities；
- `causal_gqa_attention_out_`写caller output，覆盖CPU、short fused、long hipBLASLt；
- QK提交后同一Stream把expanded K槽改写成expanded V；
- model按exact B/H/KV/T/D缓存单backing，minimum sequence默认512；
- API/CLI统计entry/hit/miss/eligible/bypass/capacity；
- CPU/HIP覆盖MHA/GQA、T1/T256、别名、零transfer和完整logits。

## 结果

Qwen/DeepSeek T512分配2895→2295、3375→2675，但吞吐仅1.004×/1.002×；peak增加
2.75/4.72MB。60/60完整logits exact。模型策略拒绝、默认关闭；out原语保留。

完整报告：[Experiment 187](../optimization-log/experiments/187-attention-core-arena-discard.md)。
