# Step 146 — typed elementwise vector16

Status: selective keep; broad policy rejected

同一个20格PyTorch ROCm矩阵依次测scalar、所有dtype vector16、选择性vector16。广泛策略让
FP32带宽格跌到0.845×–0.879×，因此不能合入。最终只对FP16/BF16、≥4M元素、三指针对齐的
调用启用，四个16M格相对scalar提升1.277×–1.411×，正确性和峰值不变。

详细记录见[Experiment 330](../experiments/330-pytorch-custom-op-vector16.md)。

