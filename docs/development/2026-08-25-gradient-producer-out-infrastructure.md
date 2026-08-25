# 2026-08-25 — caller-owned weight-gradient producer

## Problem

Experiment 259 showed that preinstalling a leaf target is insufficient: the producer allocates a
gradient Tensor and then Autograd adds it into the target. The independent question is whether the
producer can write the final address directly and remove both operations.

## Change

- add `matmul_weight_gradient_out_` for rank-2 `input^T @ output_gradient`;
- reuse the caller-owned, transpose-aware hipBLASLt matmul implementation;
- preserve caller Storage and reject wrong rank/shape/dtype/device/alias contracts;
- add CPU and MI300 tests with zero payload transfers;
- add the value to the complete PyTorch operator oracle;
- add an Event/wall benchmark comparing allocating producer plus leaf add against direct output;
- add a five-shape, three-process rotated matrix runner.

The matrix covers Model-S output head, FFN and Attention at T32, output head at T512, and a tiny
counterexample. Every process checks the complete output, logical allocations, and both operation
orders before timing.

## Pre-measurement evidence

CPU, HIP, and PyTorch parity pass. The T32 Model-S output-head pilot is bit-exact over 3,145,728
elements, removes one logical allocation per invocation, and measures 1.867x Event / 1.581x wall.
No Autograd or model route is added before the formal shape matrix.
