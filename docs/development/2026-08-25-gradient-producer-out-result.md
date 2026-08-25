# 2026-08-25 — caller-owned gradient producer结果

![Gradient producer out matrix](../optimization-log/assets/gradient-producer-out-matrix.svg)

五个shape的完整输出全部位级一致，logical allocation 1→0。Event为1.178×–1.873×，Wall为
1.101×–1.612×，CPU/HIP/PyTorch对齐。

因此准入一个scoped Autograd门，不恢复模型/DDP route。下一节点必须显式证明首个/唯一贡献、
零初始化和地址保持，无法证明时仍走普通allocating accumulation。

发布门：CPU `363/363`、ASan/UBSan `361/361`、RCCL `30/30`，producer的CPU/HIP/PyTorch
targeted parity通过，119个测试源注册。
