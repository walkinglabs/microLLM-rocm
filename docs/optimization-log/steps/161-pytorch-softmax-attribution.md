# Step 161 — typed Softmax submission attribution

Status: complete

FP16 width4096 Event为PyTorch/raw/C++/Python 4.530/4.764/4.815/5.086μs。C++/raw只有1.011×，
Python/C++为1.056×，raw/PyTorch为1.052×。下一尺度是C++ PyTorch Custom Op，不再盲调Kernel。

详细记录见[Experiment 345](../experiments/345-pytorch-softmax-attribution.md)。
