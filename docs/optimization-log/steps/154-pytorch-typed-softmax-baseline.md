# Step 154 — direct typed Softmax baseline

Status: correctness keep; performance rejected

FP16/BF16 Softmax直接写caller输出，10/10 PyTorch格、pointer和zero-temporary门通过。serial row Kernel
在width1024/4096只有约0.011×/0.004×Torch；下一步必须block-parallel reduction。

详细记录见[Experiment 338](../experiments/338-pytorch-typed-softmax-baseline.md)。
