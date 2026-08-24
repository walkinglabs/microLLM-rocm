# Step 36 — post-training-micro saturation audit

Status: complete, local track closed

## Decision

Close local training launch/cast fusion. After removing load and setup with a one-step/three-step
delta, GEMM plus AdamW own 72.71%/83.77% of Qwen/DeepSeek Kernel time. Every other individual
category has too little perfect-removal headroom, and three recent model experiments already
miss their cross-model gates.

The next accepted proposal must change GEMM grouping/algorithm selection, optimizer memory
traffic, or graph-wide lifetime/capture. Another single cast, add, repeat or Norm launch is not a
new hypothesis.
