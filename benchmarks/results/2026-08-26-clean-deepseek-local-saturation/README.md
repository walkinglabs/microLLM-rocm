# Clean DeepSeek T2048/B2/N64 local saturation

The current workload is 1.1393x PyTorch with exact tokens and lower peak memory.
Finalize alternatives are closed, grouped rows2 misses its model gate, and casts are
only 4.11% with adjacent routes already measured. Further useful work must change
architecture, workload, or hardware scope.

![Local saturation](local-saturation.svg)
