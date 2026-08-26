# INT8权重格式：从硬件probe到公共Tensor合同

日期：2026-08-26
状态：格式与反量化已验收，模型计算路由未实现

## 为什么做

仓库已经测过MI300X原始INT8矩阵硬件，但用户仍无法在框架里创建一字节有符号Tensor、保存
scale或从safetensors加载后在GPU还原。硬件峰值不能替代框架API。

## 本节点改动

- `DType::Int8`、一字节Storage、CPU/HIP transfer和非连续view逻辑读取；
- `Int8ScaledTensor = I8 values + same-device F32 scalar scale`；
- FP32/FP16/BF16到I8的CPU/HIP对称量化，以及I8到三种浮点输出的反量化；
- 最近偶数舍入、`[-127,127]`饱和、NaN归零和Inf饱和的固定语义；
- safetensors `I8`读取与`Preserve`混合I8/F32写出；
- C++↔官方Python safetensors双向互操作；
- 独立PyTorch量化oracle与MI300X逐字节/传输门；
- 初学者设计文档、算子合同和autoresearch风格边界图。

## 实测证据

- CPU聚焦5/5；
- PyTorch oracle 1/1；
- 官方safetensors互操作1/1；
- MI300X聚焦3/3；
- HIP量化对FP32/FP16/BF16输入逐字节等于CPU；
- 固定scale只产生一次4字节H2D元数据上传；
- 反量化热路径0 H2D、0 D2H；
- I8文件直接加载HIP传6字节values+4字节scale，之后无payload回传。

完整回归为CPU 415/415、ASan/UBSan 412/412、PyTorch-enabled CPU 416/416、
MI300X HIP 205/205、RCCL 55/55；覆盖清单审计包含195个Tensor/算子API、45个图API和
156个测试文件。

## 没有越界声称

本节点没有INT8 Linear、Transformer模型路由、官方Qwen/DeepSeek量化checkpoint、训练或
tokens/s。理论4×权重payload缩减不是整机显存或速度结论。下一节点是完整输出先行的
weight-only Linear，再比较显式反量化和融合/原生INT8 GEMM。
