# Qwen3-0.6B explicit BF16 inference

FFN and Attention Linear weights become one-way single-representation BF16; Q/K-Norm and other
normalization weights remain FP32. microLLM and Transformers use the same official BF16 checkpoint,
token 1 and four greedy output tokens.

Accuracy is judged against the common FP32 logits, not by requiring two different BF16 reduction
trees to be identical. microLLM is closer to FP32 than Transformers BF16 in both Max and RMS.
End-to-end generation uses two warm-ups and five measured repetitions in each framework. The
policy remains explicit and pinned to this model/evidence scope.
