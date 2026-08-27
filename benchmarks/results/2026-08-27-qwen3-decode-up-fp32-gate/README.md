# Qwen3 phase-selective decode-up FP32 complete gate

This directory decides the default-off route introduced in Experiment 375. The route uses three
BF16 FFN projections during prefill, then BF16 gate/down plus FP32 up during cached decode.

## Decision

Keep the route as an explicit precision policy. Do not make it the default and do not report it as a
speed optimization.

- shape: 64/64 fresh workers, 32 aggregate rows, 23 direct cross-framework passes and nine visible
  precision mismatches;
- KV: 24/24 cached rows have exact actual/theoretical/active bytes and zero waste;
- oracle: candidate argmax matches PyTorch FP32 in all eight fixed first-split states;
- strict common-oracle boundary: microLLM/PyTorch FP32 complete logits pass 7/8 fixed thresholds;
  T128/B1 is the pre-existing `2.346e-4 / 4.628e-5` Max/RMS boundary, while both FP32 argmax values
  and both low-precision argmax values are 320;
- performance: two independent 30-process matrices pass all 10/10 case gates; throughput geometric
  means are `0.97984x` and `0.98178x`;
- the old counterexample is overturned: T512/B2 prefill is `1.00112x`, not global up-FP32's
  `0.88751x`;
- memory: resident weights rise exactly 352,321,536 bytes (336 MiB) over current all-BF16;
  incremental engine peak is unchanged in all five measured cases.

## Reproduce

The complete shape command is saved in `shape-command.txt`. The oracle command is:

```bash
HIP_VISIBLE_DEVICES=1 /tmp/microllm-torch-rocm-venv/bin/python \
  benchmarks/single_gpu/qwen3_decode_up_fp32_oracle_sweep.py \
  --manifest /tmp/microllm-qwen3-runtime-manifest-v2.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --pytorch-python /tmp/microllm-torch-rocm-venv/bin/python \
  --output-directory /tmp/qwen3-decode-up-fp32-oracle-sweep \
  --allow-amdsmi-fallback
```

The five-case performance command is:

```bash
HIP_VISIBLE_DEVICES=2 python3 \
  benchmarks/single_gpu/compare_qwen3_decode_up_fp32_matrix.py \
  --manifest /tmp/microllm-qwen3-runtime-manifest-v2.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory /tmp/qwen3-decode-up-fp32-performance
```

Each performance case has three fresh processes per policy, alternating policy order, with two
warm-up and five measured iterations. Output changes are accepted only by the separate FP32 oracle.

## Files

- `shape-*`: 64 worker records and 32 aggregate rows;
- `oracle-summary.json`, `oracle-raw.jsonl`, `oracle-cases/`: eight complete-logit audits;
- `performance-*`: first 30 process records and five aggregate cases;
- `performance-repeat-*`: independent second 30-process matrix retained after a first-run outlier;
- `summary.json`: compact final decision.
