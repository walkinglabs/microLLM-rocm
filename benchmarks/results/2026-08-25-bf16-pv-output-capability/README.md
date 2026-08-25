# BF16 P×V output capability rejection

A temporary correctness-first candidate changed only the P×V output layout
from FP32 to BF16 while keeping FP32 probabilities, FP32 values, FP32 compute,
the existing interleaved BTHD/GQA strides and complete-output comparison.

Both the ordinary BTHD descriptor and the zero-stride GQA descriptor returned
hipBLASLt status 6 before timing. Their retained FP32 paths passed immediately
before the candidate calls. The candidate public APIs and layout-key changes
were removed. This is a backend capability rejection, not a performance result.

The two exercised contracts correspond to batch-two hand-valued conformance
fixtures; the model route was never created.
