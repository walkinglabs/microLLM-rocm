# Indexed safetensors streaming

The public index inspector validates relative shard paths and returns normalized
weight-to-shard metadata. An uninitialized HIP model verifies each tensor is in the
declared shard, then reuses transactional multi-shard streaming. BF16 H2D remains
parameter_count×2 with zero D2H.

![Index streaming](indexed-streaming.svg)
