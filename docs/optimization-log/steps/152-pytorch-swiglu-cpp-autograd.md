# Step 152 — C++ SwiGLU Autograd

Status: complete; keep/recommend

同GPU producers下，C++/Python Autograd提升1.286×–1.475×。FP32 F+B为native Torch的
1.144×/1.136×，peak仅1,536B；低精度提升到0.799×–0.812×且peak回到native。C++成为adapter
默认，低精度typed fused backward作为下一独立线。

详细记录见[Experiment 336](../experiments/336-pytorch-swiglu-cpp-autograd.md)。

