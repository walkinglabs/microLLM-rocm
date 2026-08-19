# DeepSeek-R1-Distill-Qwen-1.5B MI300X evidence

This is the 1.5B dense Qwen2.5-derived Distill model. It is not the 671B DeepSeek-R1/V3
MLA/MoE architecture.

- official BF16 checkpoint: 339 tensors and 1,777,088,000 parameters;
- strict load and FP32 reference compute on MI300X/gfx942;
- full 151,936-logit comparison passes `atol=3e-4`;
- basic DeepSeek reasoning chat prompt IDs match Transformers;
- eight greedy KV-cache tokens and decoded text match exactly;
- checkpoint itself is not committed to Git.
