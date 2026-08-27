# Qwen3 T128/B2 BF16 FFN layer search

本实验只改变哪些FFN层保留BF16；Attention和Cache固定FP32，decode输入固定，完整logit oracle
固定。runner先分半，再检查0–6单层与两个致因组内部的pair，最后对最小集合和近边界各跑3个进程。

```bash
/tmp/microllm-torch-rocm-venv/bin/python \
  benchmarks/single_gpu/qwen3_bf16_ffn_layer_search.py \
  --manifest /tmp/microllm-qwen3-runtime-manifest-v2.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --pytorch-python /tmp/microllm-torch-rocm-venv/bin/python \
  --output-directory /tmp/qwen3-bf16-ffn-layer-search-formal \
  --allow-amdsmi-fallback
```

结果：

- active 14–27保持oracle token320；active 0–13翻成25；
- active 0–2与active 3–6都能翻转；
- 0–6任意单层都不翻转；
- 组0–2的任意pair都不翻，只有三层一起翻；
- 组3–6的9个内部pair中只有`{3,4}`翻转；
- `{0,1,2}`与`{3,4}`分别3/3选择25；
- 近边界`{4,6}` margin只有0.0003157，但3/3仍选择320。

“最小”只表示在已声明组内，所有单层和对应proper pair已经检查；它不是穷举全部`2^28`
组合的数学证明。下一步应拆FFN gate/up/down子阶段，而不是继续搜索不相干的后半层。

[`summary.json`](summary.json)保存分组、single、pair和repeat门；[`raw.jsonl`](raw.jsonl)保存
28个选择性进程。base oracle的完整logit文件由runner生成但不提交Git。
