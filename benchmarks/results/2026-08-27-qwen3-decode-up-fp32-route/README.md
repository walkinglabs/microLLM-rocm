# Qwen3 decode-up FP32 phase route

This node implements and validates the route required by Experiment 374's next hypothesis. It does
not contain the complete shape or performance decision.

The route keeps:

- gate/down parameters in BF16;
- each up parameter in FP32 for cached decode;
- one derived BF16 up mirror for ordinary and cached prefill.

The phase comes from the explicit model call path. It is never inferred from `sequence == 1`, because
a valid prefill may also contain one token. Prefill can therefore pass three BF16 weights to the fused
FFN, while uniform and positions-aware cached decode use the existing readable BF16/FP32/BF16 path.

## Evidence

- CPU suite: 433/433;
- ASan/UBSan suite: 430/430;
- HIP suite: 215/215, with one unrelated multi-device test skipped because only one device was exposed
  to that process;
- CLI/runner contract: 82/82 inference-matrix tests plus binary and fixture-backed CLI tests;
- official Qwen3 smoke: 28 FP32 decode tensors, 28 BF16 prefill mirrors, 1,855,717,376 resident bytes,
  and token 25 for the fixed one-step smoke.

Relative to the current all-BF16 policy, the exact resident delta is 352,321,536 bytes (336 MiB).
Relative to global up-FP32, the BF16 prefill mirrors add 176,160,768 bytes (168 MiB). The preparation
peak remains 2,912,681,984 bytes in this smoke because the original FP32 weights already coexist with
BF16 candidates during transactional preparation.

## Reproduce the official smoke

```bash
HIP_VISIBLE_DEVICES=2 build/hip-release/apps/microllm_hf_infer \
  --config /tmp/microllm-qwen3-fixture/qwen3-0.6b/config.json \
  --weights /tmp/microllm-qwen3-fixture/qwen3-0.6b/model.safetensors \
  --tokens 1 --device hip --top-k 1 --batch 1 \
  --use-cache true --cache-prefill-mode full --decode-mode steady \
  --batch-argmax-mode device --prefill-logits last \
  --kv-cache-dtype bf16 --cache-capacity 2 --new-tokens 1 \
  --warmup 0 --steps 1 --prefill-warmup 0 --prefill-steps 1 \
  --bf16-ffn true --bf16-ffn-decode-up-fp32 true \
  --bf16-attention true --workload decode
```

The policy remains explicit and default-off. `smoke.json` proves layout, accounting and execution;
it does not prove that the route clears the eight-oracle, 64-worker or five-case performance gates.
