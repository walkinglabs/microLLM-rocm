# 2026-08-19 — Qwen2.5 architecture compatibility gate

## Pinned target

The first external model target is `Qwen/Qwen2.5-0.5B` at revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`. The tracked config fixture and external
BF16 checkpoint come from that model repository.

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

## Real checkpoint result

- 290/290 tensors load strictly on MI300X;
- fixed IDs `1,2,3,4`: 151,936 logits pass `atol=3e-4`, max abs `1.707e-4`,
  MSE `2.676e-10`, cosine `0.99999994` versus Transformers FP32;
- C++ Qwen byte-level BPE matches Transformers on 14 boundary cases;
- basic Instruct system/user/assistant rendering and all 30 chat token IDs match;
- `Hello world` maps to `[9707,1879]` in both implementations;
- four greedy KV-cache tokens match exactly: `[0,358,2776,264]`, text `! I'm a`.
- one official-weight backward/AdamW step matches PyTorch loss within `1.383e-5` and
  produces an identical observed parameter update.

Measured MI300X smoke: load 10.14 s, first two-token forward 1.44 s, four-token greedy
generation 345.54 ms, current engine memory 1.976 GB, peak 3.952 GB. Raw compact evidence is tracked under
`benchmarks/results/2026-08-19-qwen25-0.5b/`. Remaining gates are per-layer hidden-state
traces, tool-call chat rendering, BF16 compute parity, optimized loading/cache, and SFT.
