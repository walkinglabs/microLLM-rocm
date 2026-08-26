# Step 166 — official PyTorch hidden-state alignment

Status: complete

真实Qwen/DeepSeek分别覆盖27/31个embedding→blocks→norm→logits阶段。embedding exact，首差都在
block0；最终logits Max为8.01e-5/2.48e-5。

详细记录见[Experiment 350](../experiments/350-official-pytorch-hidden-alignment.md)。
