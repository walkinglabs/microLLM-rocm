# DeepSeek block-0 FP32 FFN counterfactual

This matrix tests whether the first low-precision amplifier found in Experiment
298 dominates the complete 28-layer error. It keeps BF16 KV and materialized
cached Attention fixed, then compares FP32 Linear, all 28 BF16 FFNs, and BF16
FFN with block 0 kept in FP32.

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/audit_bf16_ffn_layer_counterfactual.py \
  --manifest /path/to/pinned-two-model-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory \
    benchmarks/results/2026-08-26-deepseek-bf16-ffn-layer-counterfactual \
  --model deepseek-r1-distill-qwen-1.5b --context 2048 \
  --runs 2 --warmup 1
```

![Block-0 FP32 counterfactual](counterfactual.svg)

The conversion contract is exact: `0`, `84`, and `81` BF16 tensors for FP32,
all-BF16, and block-0-FP32. All 24 fresh processes repeat bitwise and host/device
argmax agrees; every step-0 top token is 151643.

Keeping block 0 in FP32 reduces the global maximum cross-batch Max error from
`0.0629854` to `0.0569688` (9.55%) and RMS from `0.0251711` to `0.0143833`
(42.86%). The result is not robust across batch sizes: B2 improves, but B4/B8
Max gets 12.7%/20.5% worse, and B8 RMS gets 16.3% worse. B2/B4/B8 rows are not
bitwise equal even in the FP32 Linear control.

The selective policy also retains 82,575,360 additional peak bytes and changes
median decode throughput by +0.36%, -0.24%, -0.58%, and -0.64% at B1/B2/B4/B8.
It is rejected as a precision policy. The next experiment inventories a common
BF16 hipBLASLt solution across decode rows 1/2/4/8 and tests algorithm consistency
instead of retaining more FP32 layers.

The first measurement attempt executed all 24 workers but discarded its output
before evidence publication because the summarizer requested a nonexistent
throughput field. Commit `985fe2a` fixed the contract and this directory comes
from a complete fresh rerun. `raw.jsonl` and `summary.json` are authoritative.
