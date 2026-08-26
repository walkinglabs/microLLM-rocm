# INT8 M=1融合候选

日期：2026-08-26
状态：显式候选保留，Auto不变

直接读取I8 weight和设备scale，完整Qwen/DeepSeek shape正确并少一次完整浮点weight分配。三进程
中位数相对显式反量化Event 14.09×/7.51×，但相对PyTorch常驻FP32 GEMM仅0.916×/0.494×。
因此保留`FusedDecode`供内存优先研究，默认仍走可读control；没有整模或普遍性能声明。

CPU 416/416、ASan/UBSan 413/413、PyTorch-enabled CPU 417/417和RCCL 55/55沿用紧邻基线
节点完整回归；新增融合正确性与benchmark门通过，HIP注册口径更新为208/208。
