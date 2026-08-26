# DeepSeek step-0 cross-batch precision-island isolation

This matrix keeps BF16 KV and the same auto-enabled materialized Attention path.
Only Linear weight preparation changes: FP32 Linear, BF16 FFN only, BF16
Attention only, or both BF16 islands. B1 is the complete-logit reference.

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/audit_cross_batch_precision.py \
  --manifest /path/to/pinned-two-model-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory \
    benchmarks/results/2026-08-25-deepseek-cross-batch-precision \
  --model deepseek-r1-distill-qwen-1.5b --context 2048 \
  --runs 2 --warmup 1
```

![Precision-island isolation](precision-isolation.svg)

All 32 processes are deterministic and host/device argmax agrees. FP32 Linear
already has a small cross-batch Max/RMS drift of 0.001354/0.000229. BF16
Attention-only raises it to 0.020970/0.004278. BF16 FFN-only reaches
0.062985/0.025171, making FFN the primary amplifier. The combined current path
reaches 0.067570/0.017350.

Every step-0 top token remains 151643, so this matrix locates numerical drift
rather than a sampling failure. The next experiment traces BF16 FFN block
boundaries to find the first layer where B1 versus B2/B4/B8 error expands.
No scheduler default changes.

`raw.jsonl` and `summary.json` are authoritative. `analysis.json` records the
inference and `verification.json` pins the clean commit, exact converted-tensor
counts, and full gates.
