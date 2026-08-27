# Qwen3 global gate-FP32 calibration rejection

候选让所有FFN gate保持FP32，up/down与全部Attention为BF16，Cache为BF16。它不是手工层组合，
而是一条全模型、可解释、shape无关的简单校准规则。

```bash
/tmp/microllm-torch-rocm-venv/bin/python \
  benchmarks/single_gpu/qwen3_bf16_gate_fp32_oracle.py \
  --manifest /tmp/microllm-qwen3-runtime-manifest-v2.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --pytorch-python /tmp/microllm-torch-rocm-venv/bin/python \
  --output-directory /tmp/qwen3-bf16-gate-fp32-oracle-v2 \
  --allow-amdsmi-fallback
```

| case | FP32 | candidate | Transformers BF16 | candidate匹配 |
|---|---:|---:|---:|---|
| T32/B1 step1 | 374 | 374 | 323 | 是 |
| T32/B2 step1 | 374 | 374 | 323 | 是 |
| T128/B2 step8 | 320 | 320 | 320 | 是 |
| T512/B1 step2 | 2955 | 1096 | 1096 | 否 |
| T512/B2 step8 forced | 1273 | 1273 | 4285 | 是 |

T512/B1候选错误margin为0.003286，说明保留gate FP32并不能普遍保住top-2顺序。4/5不能通过
预先固定的“全部case匹配FP32”门，因此不运行32-row或性能矩阵。

[`summary.json`](summary.json)保存五格oracle；[`raw.jsonl`](raw.jsonl)保存20个worker。
