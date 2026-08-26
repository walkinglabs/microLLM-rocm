# Official Qwen output-column INT8 rebuttal

The same pinned Qwen token-1 gate compares scalar and per-output-column weight scales. Column
scales improve complete-logit Max/RMS and recover the second generated token, but the first argmax
and token remain wrong by a wide margin. Qwen fails before DeepSeek is run, following the fixed
stop rule.

The column quantization/dequantization and fused M=1 primitives remain explicit APIs. The current
weight-only INT8 model-precision line is closed; no default or official inference claim exists.
