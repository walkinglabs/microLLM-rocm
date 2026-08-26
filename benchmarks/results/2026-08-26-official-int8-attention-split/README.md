# Final official Qwen INT8 Attention split

The final permitted PTQ decomposition compares QKV-only and O-projection-only output-column INT8.
Both preserve two generated tokens and improve throughput, but both miss the fixed complete-logit
Max/RMS limits. The limits are not relaxed after observing the result.

This closes the current official weight-only INT8 PTQ line. Primitive APIs and explicit scopes
remain for teaching and future calibrated/QAT work; none becomes Auto or default.
