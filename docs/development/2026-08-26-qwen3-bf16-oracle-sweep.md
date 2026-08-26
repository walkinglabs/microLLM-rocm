# Qwen3 BF16全部分叉的oracle sweep

日期：2026-08-26
状态：5个去重case、8个矩阵row全部归因

新增：

- `audit_qwen3_bf16_divergence.py`支持B2、batch内完整logit误差、policy子集和固定decode输入；
- `qwen3_bf16_oracle_sweep.py`固定5个首次分叉合同并聚合28个worker；
- `microllm_hf_infer --forced-decode-inputs`只在单次zero-warmup steady cached诊断中启用；
- CLI拒绝错误模式、数量和词表外token；JSON记录forced状态与数量。

结果为microLLM mixed/Transformers BF16匹配FP32 case 4/1、矩阵row 7/1。T128/B2是明确
microLLM反例，不能删除。T512/B2使用共同9-token输入，避免把不同自然轨迹的logits硬比较。

这个节点没有训练或速度结论。证据目录保存5-case summary与28行raw；测试固定case常量、winner、
argmax、forced输入和CLI负例。CPU 431/431与MI300X HIP 214/214全量回归通过；forced CLI的
ASan/UBSan binary与真实tiny batch路径2/2通过。
