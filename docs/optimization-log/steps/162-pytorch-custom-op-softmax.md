# Step 162 — C++ PyTorch Custom Op Softmax

Status: integration keep; wide partial

CPU/ROCm/Meta/C++ Autograd注册完成。no-grad门让FP16 width4096 Custom Op提升1.158×；width1024
FP16/BF16达到1.026×/0.993×native，width4096仍只有0.795×/0.529×。

详细记录见[Experiment 346](../experiments/346-pytorch-custom-op-softmax.md)。
