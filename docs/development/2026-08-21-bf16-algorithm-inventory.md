# BF16 M32/M64 hipBLASLt algorithm inventory

An independent CLI now queries hipBLASLt with the same BF16 row-major descriptor mapping as the
engine. It reports solution index, required workspace and waves for each shape plus their set
intersection; it does not change default dispatch.

On the fixed MI300X/ROCm environment, both M32 and M64 gate shapes return 64 candidates and share
53 indices under a 32 MiB limit. A same-algorithm counterfactual is therefore feasible. Indices are
explicitly version-local.

See [Experiment 109](../optimization-log/experiments/109-bf16-algorithm-inventory.md).
