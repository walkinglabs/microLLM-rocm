# Qwen2.5-0.5B MI300X evidence

- official BF16 checkpoint, 290 tensors, 494,032,768 parameters;
- microLLM readable FP32 compute on MI300X/gfx942;
- Transformers CPU FP32 compute using the same decoded BF16 weights;
- fixed token full-logit comparison passes `atol=3e-4`;
- Qwen byte-level BPE matches on English, whitespace/newline, numeric, contraction, and
  Chinese cases;
- prompt `Hello world` and four greedy KV-cache tokens match exactly.

The checkpoint is intentionally not stored in Git. Reproduction uses the pinned model
revision recorded in `comparison.json`, `microllm_hf_infer`,
`pytorch_qwen_reference.py`, and `compare_logits.py`.
