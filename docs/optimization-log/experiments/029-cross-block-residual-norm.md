# Experiment 029 — cross-block residual + RMSNorm

Status: `discard`

Cached inference was rearranged so each block's second residual add and the next block's
Attention RMSNorm used the existing pair-output fused operator; the last block fused with
final Norm. Focused cache, graph and token tests passed and 28 launches were removed.

The first official matrix returned Qwen `209.54` (-4.4%) and DeepSeek `79.57` (+1.1%)
token/s; score fell to `2.456886`. The hard gate failed, so the scheduling rewrite was
removed. Raw evidence is in [029-data](029-data/README.md).
