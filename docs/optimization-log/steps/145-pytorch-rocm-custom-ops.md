# Step 145 — PyTorch ROCm Custom Op完整门

Status: complete; integration keep, speedup rejected

先让可选Torch adapter在Torch 2.11 ROCm/HIP 7.13上真实构建，再扩展FP32/FP16/BF16
caller-owned elementwise、Autograd公式和Meta dispatch。六进程矩阵轮换Torch/microLLM先后，
每格5次热身、25次测量。

20/20数值格完全一致，当前Stream、所有权、Autograd与fullgraph compile均通过；但Event中位数
全部低于Torch，范围0.469×–0.973×。因此集成能力保留，elementwise加速说法拒绝。下一步只允许
向量化typed Kernel或更大融合边界。

详细记录见[Experiment 329](../experiments/329-pytorch-rocm-custom-ops.md)。

