# Qwen3 BF16第一次分叉归因

日期：2026-08-26
状态：T32/B1第一次分叉已归因

新增`audit_qwen3_bf16_divergence.py`，固定相同prompt、Cache语义和capture step，运行：

- microLLM FP32 weight + FP32/BF16 Cache；
- microLLM BF16 weight + FP32/BF16 Cache；
- Transformers FP32/BF16。

runner保存六条worker记录和完整logit统计。两种FP32实现Max/RMS为
`5.78e-5/1.20e-5`，共同选token374。整网BF16将374/323舍入为相同14.1875并选323；
microLLM mixed BF16保留0.03126 margin并选374。

这一步证明T32/B1失败来自精度policy遇到低margin，不授权删除矩阵中的失败，也没有性能结论。
测试固定六个policy、151,936元素、FP32误差门、top tie、argmax和raw行数。
