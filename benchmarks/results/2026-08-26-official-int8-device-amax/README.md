# Official Qwen INT8 device-amax gate

The same pinned Qwen fixture, token 1 and MI300X process compare FP32 with explicit whole-model
INT8. Device preparation scans every Linear without weight D2H. Raw prefill logits are complete
151,936-element FP32 files and are excluded from Git; the committed summary records their full
Max/RMS and argmax comparison.

Preparation and residency pass, short decode is faster, but complete logits and generated tokens
fail decisively. The official-model route is rejected. Device-only dynamic quantization and the
explicit Model-S route remain; no default changes.
