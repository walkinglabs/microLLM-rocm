# Step 163 — caller-owned PyTorch Softmax out

Status: integration keep; wide partial

`Tensor(a!)` alias/mutation与inference-only边界完成。10格pointer/精度/零peak通过；width1024
FP16/BF16为1.116×/1.087×native out，width4096仍为0.813×/0.467×。

详细记录见[Experiment 347](../experiments/347-pytorch-custom-op-softmax-out.md)。
