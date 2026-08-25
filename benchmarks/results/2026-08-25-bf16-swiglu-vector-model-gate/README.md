# BF16 vectorized SwiGLU full-model gate

The retained B1T1024 grouped-FFN/BTHD inference policy compares the prior scalar
binary with a candidate whose Auto route uses vectorized BF16 SwiGLU. Each model
and policy runs in three fresh processes with two warm-ups and five measurements.

Qwen improves 1.0073x. DeepSeek improves only 1.0005x and fails the predeclared
1.005x gate. Both complete vocab outputs are bit-identical; peak bytes and engine
allocation calls are unchanged. Auto therefore remains scalar while the explicit
operator is retained.
