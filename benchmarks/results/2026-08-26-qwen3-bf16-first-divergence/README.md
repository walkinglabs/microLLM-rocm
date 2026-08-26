# Qwen3 BF16 第一个 token 分叉

Experiment 364 的最短稳定失败是 T32/B1/N4：两边第一个 steady decode token 都是1，
第二个 token 变成 microLLM 374、Transformers 323。本目录在该选择发生前导出全部151,936
logits，并以 Transformers FP32 为独立 oracle。

```bash
/tmp/microllm-torch-rocm-venv/bin/python \
  benchmarks/single_gpu/audit_qwen3_bf16_divergence.py \
  --manifest /tmp/microllm-qwen3-runtime-manifest-v2.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --pytorch-python /tmp/microllm-torch-rocm-venv/bin/python \
  --output-directory /tmp/qwen3-bf16-divergence-runner \
  --context 32 --decode-tokens 4 --capture-step 1 \
  --allow-amdsmi-fallback
```

六条policy分别是两种FP32实现、microLLM权重/Cache的四格组合，以及Transformers整网BF16。
microLLM FP32对oracle Max/RMS为`5.78e-5/1.20e-5`，并与oracle生成相同token。

FP32 oracle中token374/323为`14.14219/14.10959`，margin只有`0.03260`。Transformers BF16
把两项都舍入为`14.1875`，形成exact tie，并由argmax选较小索引323。microLLM mixed BF16
为`14.15291/14.12165`，仍保留`0.03126` margin并选择FP32 token374。

因此本固定失败由两条精度policy遇到低margin共同触发，不是microLLM token错误。这个结论只覆盖
T32/B1的第一次分叉；它不删除Experiment 364的8个公开precision limits，也不证明所有microLLM
长轨迹都更接近FP32。

[`raw.jsonl`](raw.jsonl)保存六个worker合同；[`summary.json`](summary.json)保存完整logit误差、
top3、margin、weight/cache四格归因和所有判定门。完整logit二进制由runner生成但不提交Git。
