# 2026-08-19 — DeepSeek-R1-Distill-Qwen-1.5B

## Sequence of work

1. Read the pinned official config and confirmed `model_type=qwen2`.
2. Added a tracked config fixture and exact 1,777,088,000-parameter gate.
3. Extracted the official tokenizer vocabulary/merges and identified Distill-specific
   begin/User/Assistant/think special tokens.
4. Implemented the basic reasoning chat template and matched all 12 prompt token IDs.
5. Strictly loaded all 339 BF16 checkpoint tensors on MI300X.
6. Compared every final logit with Transformers FP32.
7. Compared eight greedy KV-cache tokens and decoded text.
8. Recorded loading/forward/generation time and engine memory.

## Result

Full-logit max absolute difference is `6.409e-5`; eight generated IDs are identical.
The compact evidence directory is
`benchmarks/results/2026-08-19-deepseek-r1-distill-qwen-1.5b/`.

## Boundary

This closes the dense 1.5B Distill target only. It does not implement flagship
DeepSeek-R1/V3 MLA, MoE, expert parallelism, or their FP8 policy.
