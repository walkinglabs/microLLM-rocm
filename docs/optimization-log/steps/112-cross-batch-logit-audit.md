# Step 112 — DeepSeek cross-batch complete-logit audit

Status: complete; batch-shape numerical drift confirmed

Experiment 294中DeepSeek跨框架token在B1/B8从index 2分叉，B2/B4相同。下一步先排除microLLM自身
batch语义错误：

1. 固定同一T2048 prompt、BF16 KV、no-flag Auto；
2. 对B1/B2/B4/B8分别导出decode step 0/1/2的完整logits；
3. 每个batch所有行必须彼此相同，并与B1比较Max/RMS/top1；
4. 同时保存host/device argmax对照，排除采样路径；
5. 每种batch至少两个fresh process，检查确定性；
6. 若micro内部一致，则把分叉归类为precision-policy边界并补同政策PyTorch对照；
7. 若micro内部不一致，从第一处分叉层做trace，不先调scheduler。

在该审计通过前，不设置模型特定batch默认。

## 实测结果

24进程全部确定；batch内行位级相同；host/device argmax全相同。跨batch从step0即不同，最终Max/RMS
0.197803/0.046133；step2 B1/B8 token151643，B2/B4 token3555。排除行混写和argmax，Step 113
隔离FP32/BF16 FFN/BF16 Attention四种精度路径。
