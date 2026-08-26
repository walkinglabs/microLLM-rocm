# Step 158 — FP16-only wave reduction

Status: complete; selective keep

cached FP16 width2048–8192编译wave reduction，BF16编译shared-tree fallback。width4096 FP16
Event/wall提高1.077×/1.080×，BF16为1.002×/1.004×基线。当前FP16仍只有0.615×PyTorch。

详细记录见[Experiment 342](../experiments/342-pytorch-fp16-wave-softmax.md)。
