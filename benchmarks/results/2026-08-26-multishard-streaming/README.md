# Multi-shard safetensors streaming

An uninitialized HIP model now preflights every shard header as one transaction,
allocates one bounded staging tensor per source dtype, then visits one shard at a
time. BF16 H2D bytes equal parameter_count×2; D2H and D2D are zero. Missing or
duplicate tensors fail before any H2D copy.

![Streaming design](multishard-streaming.svg)
