# Folded GQA GEMM candidate rejected

The candidate folded query heads sharing one K/V head into GEMM's row dimension and removed
physical K/V repetition. Focused CPU/HIP tests passed and the T2048 B8 three-process medians gained
4.3% on Qwen and 7.4% on DeepSeek while peak memory fell about 3%.

The official full-logit oracle rejected it. Against the independently built `ef6fe1e` reference,
Qwen reached max-abs/RMSE 0.0735/0.0157 and DeepSeek 0.0563/0.0119. Top tokens stayed equal, which
is specifically why top-1 cannot be the acceptance gate.

The source candidate was removed. Raw performance and precision evidence remain in
[Experiment 078](../optimization-log/experiments/078-folded-gqa-gemm-discard.md).
