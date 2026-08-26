# Step 164 — BF16 cached wave1024

Status: complete; keep

BF16 width4096 core Event/wall提高1.687×/1.578×，Custom out约1.687×；当前分别达到0.888×PyTorch
和0.804×native out。旧256-thread broad-wave拒绝仍有效。

详细记录见[Experiment 348](../experiments/348-pytorch-bf16-wave1024-softmax.md)。
