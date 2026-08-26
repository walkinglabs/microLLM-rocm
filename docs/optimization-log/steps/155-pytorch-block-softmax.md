# Step 155 — typed Softmax block reduction

Status: shape-aware keep

FP16/BF16的width≤32继续使用serial row；更宽的行改为64/128/256线程共同归约。10格PyTorch与
零临时量门通过，width128/1024达到1.21×–1.25×/1.10×–1.11×Torch；width4096提高约
146×–149×但仍只有0.43×–0.46×Torch。

详细记录见[Experiment 339](../experiments/339-pytorch-block-softmax.md)。
