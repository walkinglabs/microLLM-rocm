# Qwen3 FFN layers0–4 FP32 candidate rejection

候选保留layers0–4完整FFN为FP32，其余23层FFN与全部Attention为BF16，Cache为BF16。
目标是在保留大多数融合层的同时打断两个最小错误组合。

```bash
/tmp/microllm-torch-rocm-venv/bin/python \
  benchmarks/single_gpu/hf_inference_shape_matrix.py \
  --manifest /tmp/microllm-qwen3-runtime-manifest-v2.json \
  --micro-binary build/hip-release/apps/microllm_hf_infer \
  --pytorch-python /tmp/microllm-torch-rocm-venv/bin/python \
  --output-directory /tmp/qwen3-ffn0-4-fp32-shape-matrix \
  --models qwen3-0.6b --contexts 1,32,128,512 --batches 1,2 \
  --decode-lengths 1,4,32 --cases prefill,cached \
  --micro-kv-cache-dtype bf16 --micro-cache-capacity exact \
  --micro-bf16-ffn-fp32-layers 0,1,2,3,4 \
  --warmup 1 --steps 1 --runs 1 --allow-amdsmi-fallback
```

候选把T128/B2/N32与PyTorch的共同前缀从8延长到22，并在第9个token恢复FP32 oracle 320；
但第23个token再次分叉。T512/B2/N32更严重：两个相同microLLM batch row生成不同token，matrix
worker失败。该失败另起3个进程复现，3/3得到相同错误。

常驻按预期增加94,371,840字节（约90MiB），没有出现重复权重；但正确性先失败，所以没有运行
原计划的3×2+5性能门，相关未使用runner也被删除。

[`matrix-summary.json`](matrix-summary.json)保存完整32-row聚合，
[`raw.jsonl`](raw.jsonl)保存64个worker， [`summary.json`](summary.json)保存审查结论。
