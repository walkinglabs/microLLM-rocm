# Grouped gate/up Swish epilogue capability

This matrix enables `HIPBLASLT_EPILOGUE_SWISH_EXT` only on the gate member of
the exact B1T1024 grouped gate/up problem. Two official shapes, three fresh
processes, 64 algorithms, complete BF16 outputs and Event/wall timing are used.

All 64 candidates pass in every process. Pointer-stable user-argument medians
are 1.097x for Qwen and 1.069x for DeepSeek against the separate-GEMM capability
baseline. Reinitializing remains slower in both cases. This admits only a
full-model experiment, not a default policy.
