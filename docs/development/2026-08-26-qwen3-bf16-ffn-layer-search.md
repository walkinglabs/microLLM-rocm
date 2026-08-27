# Qwen3 BF16 FFN层组合搜索

日期：2026-08-26
状态：两个最小已验证组合

新增`qwen3_bf16_ffn_layer_search.py`，验证BF16 active层数与转换Tensor数`3×layers`，并执行
分半、0–6单层、9个内部pair以及三组3-process repeat。

结果：无单层翻转；`{0,1,2}`必须三层共同出现；`{3,4}`是唯一翻转pair；`{4,6}`是稳定但极近
边界的反例。28行raw和完整summary进入结果目录，Python evidence固定所有门。

默认策略未改变，且没有性能结论。下一步需要FFN投影级选择能力。
