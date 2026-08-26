# Step 150 — SwiGLU Autograd提交归因

Status: complete

native/custom/manual三路保持相同loss和两份梯度。manual fused比custom Autograd快4.855×–5.271×，
比native快3.859×–4.105×。剩余差距被定位到Python register_autograd/engine提交边界，数学Kernel
局部线关闭。

详细记录见[Experiment 334](../experiments/334-pytorch-swiglu-autograd-attribution.md)。

