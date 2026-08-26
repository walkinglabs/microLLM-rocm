# Qwen3 T128/B2 BF16 weight-island isolation

Oracle sweep中唯一microLLM mixed-BF16反例是T128/B2 step8。本实验固定相同9个decode输入，
把microLLM权重拆成FP32、FFN-only BF16、Attention-only BF16和两种完整组合；Cache先固定FP32。

```bash
/tmp/microllm-torch-rocm-venv/bin/python \
  benchmarks/single_gpu/audit_qwen3_bf16_divergence.py \
  --manifest /tmp/microllm-qwen3-runtime-manifest-v2.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --pytorch-python /tmp/microllm-torch-rocm-venv/bin/python \
  --output-directory /tmp/qwen3-bf16-t128-weight-islands-forced \
  --context 128 --batch 2 --decode-tokens 9 --capture-step 8 \
  --forced-inputs 14582,1,374,264,3491,429,374,537,264 \
  --micro-policies \
micro-fp32-fp32,micro-ffn-bf16-fp32,micro-attention-bf16-fp32,micro-bf16-fp32,micro-bf16-bf16 \
  --allow-amdsmi-fallback
```

| policy | argmax | top1−top2 | oracle Max/RMS | batch内Max |
|---|---:|---:|---:|---:|
| FP32 | 320 | 0.06830 | 1.43e-4 / 2.86e-5 | 6.15e-5 |
| FFN-only BF16 | 25 | 0.03416 | 0.4550 / 0.0967 | 0.1443 |
| Attention-only BF16 | 320 | 0.07367 | 0.1649 / 0.0352 | 0.0325 |
| FFN+Attention BF16, FP32 Cache | 25 | 0.05462 | 0.3457 / 0.0705 | 0 |
| FFN+Attention BF16, BF16 Cache | 25 | 0.00674 | 0.1165 / 0.0249 | 0 |
| Transformers BF16 | 320 | 0.4375 | 0.8898 / 0.1952 | 0 |

FFN-only已经足以翻转320/25；Attention-only不会。BF16 Cache能降低完整误差并缩小错误margin，
却不能恢复argmax。因此下一实验只搜索FFN层，不再修改Cache或Attention。

[`summary.json`](summary.json)保存7个policy完整统计；[`raw.jsonl`](raw.jsonl)保存worker合同。
本结果没有性能计时，也不把T128反例推广成所有context的FFN结论。
