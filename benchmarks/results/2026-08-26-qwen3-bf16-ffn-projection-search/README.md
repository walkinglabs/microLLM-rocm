# Qwen3 BF16 FFN projection search

固定两个最小层集合`{0,1,2}`与`{3,4}`，分别测试gate/up/down单独BF16、三种pair和三投影
全BF16。其余层、Attention和Cache保持FP32，输入与完整logit oracle固定。

```bash
/tmp/microllm-torch-rocm-venv/bin/python \
  benchmarks/single_gpu/qwen3_bf16_ffn_projection_search.py \
  --manifest /tmp/microllm-qwen3-runtime-manifest-v2.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --pytorch-python /tmp/microllm-torch-rocm-venv/bin/python \
  --output-directory /tmp/qwen3-bf16-ffn-projection-search-formal \
  --allow-amdsmi-fallback
```

两个集合结果相同：

- gate-only、up-only、down-only全部选择oracle token320；
- gate+up、gate+down、up+down全部选择320；
- 只有gate+up+down三者同时BF16才选择25。

`{0,1,2}`六个partial scope margin为0.0531–0.1175；`{3,4}`为0.0343–0.1118。
如果只按当前case追求最大安全margin，保留gate FP32、让up+down BF16最好；但它是否在完整shape
矩阵和速度上合适尚未测量，不能在本节点改默认。

核心API新增`Bf16FfnWeightScope`七种显式scope。all-BF16继续走融合FFN；partial scope使用三个
Linear的可读fallback，不允许伪装成融合性能。

[`summary.json`](summary.json)保存两个case和12个scope；[`raw.jsonl`](raw.jsonl)保存20个worker。
