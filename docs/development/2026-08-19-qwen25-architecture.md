# 2026-08-19 — Qwen2.5 architecture compatibility gate

## Pinned target

The first external model target is `Qwen/Qwen2.5-0.5B` at revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`. The tracked config fixture comes from that
model repository; no real checkpoint result is claimed in this milestone.

## Implemented architecture differences

- strict Hugging Face `config.json` loader for `model_type=qwen2`;
- vocabulary 151,936, dimension 896, 24 layers, 14 Q heads, and 2 KV heads;
- FFN dimension 4,864, context 32,768, RoPE base 1,000,000;
- configurable RMSNorm epsilon, set to `1e-6` by the pinned config;
- Q/K/V bias parameters, forward Kernel, backward reduction, names, and weight mapping;
- split-half Qwen/Llama RoPE alongside the existing interleaved teaching layout;
- tied embeddings and BF16 source metadata;
- CLI inspection without constructing the 0.5B model.

The official safetensors header contains 290 tensors and Q/K/V bias tensors on every
layer. The parsed model parameter count is exactly `494,032,768`, matching the model
repository's BF16 checkpoint metadata; its BF16 weight payload is 988,065,536 bytes.

## Evidence gates

- hand-valued CPU bias forward/reduction;
- device-native MI300X bias forward/reduction with no hidden payload transfer;
- split-half RoPE CPU/HIP forward and backward;
- PyTorch forward/backward oracle for both additions;
- unsupported family, sliding-window, MRoPE, malformed scale, and shape rejection;
- `microllm_hf_inspect --config ...` integration test.

## Still required before “Qwen inference works”

- load the complete official checkpoint on MI300X;
- compare fixed-token per-layer states and logits with Transformers;
- implement the Qwen byte-level BPE tokenizer and special tokens;
- compare KV-cache prefill/decode and greedy token output;
- record memory and latency.
