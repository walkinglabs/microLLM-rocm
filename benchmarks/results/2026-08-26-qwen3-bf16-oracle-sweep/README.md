# Qwen3 BF16 全部分叉的 FP32 oracle sweep

Experiment 364 的8个 mismatch row只有5个不同的“首次分叉状态”：N4/N32有些共享同一前缀和
capture step。本结果使用统一runner逐个导出完整151,936 logits。

```bash
/tmp/microllm-torch-rocm-venv/bin/python \
  benchmarks/single_gpu/qwen3_bf16_oracle_sweep.py \
  --manifest /tmp/microllm-qwen3-runtime-manifest-v2.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --pytorch-python /tmp/microllm-torch-rocm-venv/bin/python \
  --output-directory /tmp/qwen3-bf16-oracle-sweep \
  --allow-amdsmi-fallback
```

五个去重case全部通过共同输入、FP32实现对齐和唯一低精度winner门：

| case | FP32 | micro mixed BF16 | Transformers BF16 | 匹配FP32 |
|---|---:|---:|---:|---|
| T32/B1 step1 | 374 | 374 | 323 | microLLM |
| T32/B2 step1 | 374 | 374 | 323 | microLLM |
| T128/B2 step8 | 320 | 25 | 320 | Transformers |
| T512/B1 step2 | 2955 | 2955 | 1096 | microLLM |
| T512/B2 step8 | 1273 | 1273 | 4285 | microLLM |

映射回原矩阵，microLLM mixed BF16在7/8 mismatch row匹配FP32 argmax，Transformers BF16在
1/8匹配。T128反例说明不能用“全局Max/RMS更小”推断top-2顺序一定正确。

T512/B2中，两种低精度在step2已经共同离开自然FP32轨迹；step8比较使用固定输入
`[14582,198,262,1096,374,279,2038,374,264]`。C++ CLI与PyTorch worker都明确记录
forced input count=9，确保比较同一状态。

[`summary.json`](summary.json)保存5个case与8行映射；[`raw.jsonl`](raw.jsonl)保存28个worker
合同。完整logit二进制由各case runner生成但不提交Git。mismatch状态继续保留：oracle归因回答
“哪个argmax匹配FP32”，不让两个不同BF16 policy变成数值相同。
