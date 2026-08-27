# Qwen3 gate/up全量训练对齐

日期：2026-08-27
状态：FP32通过，BF16拒绝

新增诊断API在AdamW前导出梯度、一步后导出参数；C++与PyTorch统一内部名字和`[K,N]` shape。
官方Qwen3比较56个Tensor、176,160,768个梯度元素和同量参数。

FP32 Gradient Max/RMS为3.109e-4/4.377e-7，Parameter为1.996e-5/5.645e-8，固定门全过。
BF16 Gradient为0.25356/3.597e-4，Parameter为2.003e-5/2.626e-6；Gradient Max和Parameter RMS
失败。最坏梯度在block6 up projection。

导出是diagnostic，不作性能数据。当前只覆盖gate/up；下一步扩attention/down/norm/embedding/output，
之后才能称完整官方模型梯度对齐。

回归证据：CPU 434/434、ASan/UBSan 431/431；coverage inventory为199 ops、45 graph API、159 tests。
