# Autograd external gradient pool matrix

This directory is the immutable evidence package for the explicit leaf-gradient buffer gate.

- `raw.jsonl`: 18 fresh-process records; Tiny T8 and Model-S T8/T32, both policy orders,
  three runs per order;
- `summary.json`: correctness, address, timing, logical-allocation and measured-peak medians;
- `verification.json`: matrix completeness and exactness checks.

Each process runs one warmup and five measured backward steps on the visible MI300X. Timing covers
`zero_grad + forward + backward`; gradient copies used for verification happen after timing.
The speed ratio is `baseline / external`, so values below `1.0` mean the external pool is slower.

The result is deliberately split in two: correctness passes, model-policy performance fails.
The API remains useful when a foreign runtime or communication layer requires stable addresses,
but it is not enabled as the ordinary engine training path.

