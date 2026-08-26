# Step 147 — PyTorch fused SwiGLU Custom Op

Status: API/large-forward keep; training performance rejected

注册FP32/FP16/BF16 SwiGLU forward、Meta与Autograd。六进程15格全部通过精度门。16M forward
达到1.142×–1.570×，allocator peak减半；1M forward+backward仅0.615×–0.761×，因此不做训练
加速声明，下一节点隔离backward。

详细记录见[Experiment 331](../experiments/331-pytorch-custom-op-swiglu.md)。

