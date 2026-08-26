# Step 156 — bounded shared exponential cache

Status: keep; parity open

width2048–8192的typed Softmax在LDS保留FP32 exponential，避免写回时再次调用`expf`。width4096
BF16/FP16 Event提高1.244×/1.217×，但仍只有0.550×/0.576×PyTorch；下一节点只测试wave-level
reduction。

详细记录见[Experiment 340](../experiments/340-pytorch-cached-softmax.md)。
