# N5 — 训练、生成与统一评测

## 不要先挑一条漂亮生成

A training loss alone cannot establish validation quality. Before running, record:

```text
dataset/version/license
split boundary
tokenizer and vocabulary
model config and seed
batch/context/dtype
prompt set and generation seeds
```

The engine has byte/BPE/Hugging Face tokenizers and a pinned TinyStories smoke source.
The retained run is intentionally short; there is still no reference-length Model-S
quality curve or real instruction-corpus SFT claim.

## Model-S executable evidence

```bash
"$MICROLLM_ENGINE_DIR/build/cpu-debug/examples/microllm_model_s_smoke"
"$MICROLLM_ENGINE_DIR/build/cpu-debug/examples/microllm_model_s_train_smoke"
```

Observed three-step CPU smoke:

```text
11.2473 → 5.39479 → 1.98712
```

This proves all 15.6M parameters, gradients, AdamW moments, and updates connect. It is
not a pretraining curve.

## HIP forward and training

```bash
ctest --test-dir "$MICROLLM_ENGINE_DIR/build/hip-release" -L hip --output-on-failure
"$MICROLLM_ENGINE_DIR/build/hip-release/examples/microllm_model_s_hip_smoke"
"$MICROLLM_ENGINE_DIR/build/hip-release/examples/microllm_tiny_hip_train"
```

On MI300X/gfx942, the historical Model-S CPU/HIP maximum logit error was
`4.05312e-06`. Tiny HIP training loss went from `2.21512` to `1.11681` in five steps.
The complete tiny graph and the AdamW payload update now remain device-native; explicit
metric reporting can still copy selected values to the host. N6 profiles the actual
timeline instead of guessing where time goes.

## KV Cache correctness

For MHA and GQA, every cached prefix logit is compared with full-prefix logits:

```bash
ctest --test-dir "$MICROLLM_ENGINE_DIR/build/cpu-debug" \
  --output-on-failure -R CachedLogits
```

Tolerance is `2e-5`; cache stores actual per-layer projected K/V, not token IDs.

## Generation

The generator supports greedy, temperature, top-k, and fixed seed:

```bash
ctest --test-dir "$MICROLLM_ENGINE_DIR/build/cpu-debug" \
  --output-on-failure -R 'SamplingTest|GeneratorTest'
```

## SFT response masking

```bash
"$MICROLLM_ENGINE_DIR/build/cpu-debug/examples/microllm_tiny_sft"
```

Targets whose predicted token is still inside the prompt are set to `-100` and
ignored by CPU/HIP cross entropy and backward. Observed response-only loss:

```text
1.88494 → 0.439716 → 0.0549371 → 0.0106737
```

## 预训练与 SFT 的边界

SFT uses the same optimizer and masked next-token loss. Tiny response masking is now
measured. What is still missing is a licensed/versioned instruction corpus, documented
split, Model-S SFT run, and fixed evaluation prompts. Until those artifacts exist,
Model-S SFT remains an interface capability, not a quality result. N9 separately
checks official Qwen and DeepSeek Distill weights; it does not turn a one-step smoke
into a complete fine-tuning report.

## 下一步

N6 asks where time goes. It separates Event kernel time, synchronized wall time,
host copies, allocations, and complete tokens/s before accepting an optimization.
