# INT8 weight matmul正确性基线

日期：2026-08-26
状态：公共基线已验收，优化实现未开始

新增`int8_weight_matmul(input, weight)`，只接受连续二维浮点`[M,K]`与I8`[K,N]`+F32 scalar
scale。当前实现显式还原完整浮点weight后调用普通matmul。它不是性能实现，而是后续融合候选
必须匹配的唯一公共答案和临时内存control。

聚焦证据：CPU三dtype完整输出通过；MI300X HIP输出等于CPU且测量窗0 D2H；PyTorch独立
oracle的`3×5`输出逐项相等；覆盖清单已纳入新API。

节点回归为CPU 416/416、ASan/UBSan 413/413；上一节点完整PyTorch/HIP/RCCL矩阵保持全绿，
新增PyTorch完整输出门和HIP baseline门分别通过，当前注册口径为417/417、206/206、55/55。
