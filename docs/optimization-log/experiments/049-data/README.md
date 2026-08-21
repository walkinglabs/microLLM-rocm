# Experiment 049 evidence

- `vector4-mirror.jsonl`: 12 exact element counts × scalar/vectorized, 10 Event repeats.
- `vector8-mirror.jsonl`: width-8 counterexample on the same 12 counts.
- `rsqrt-mirror.jsonl`: corrected rsqrt experiment; the initial NaN was caught by CTest.
- `vector4-no-mirror.jsonl`: embedding/output-sized tensors without BF16 mirror writes.
- `qwen-pilot/`: four shapes × one fresh microLLM/PyTorch pair using explicit Vectorized.
- `comparison.json`: exact policy gate and chart input.

`Auto` deliberately remains the scalar implementation. `Vectorized` is retained only as
an explicit operator implementation and benchmark target, not as a model speed claim.
