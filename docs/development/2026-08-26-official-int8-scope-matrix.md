# 官方INT8混合精度scope矩阵

日期：2026-08-26
状态：FFN拒绝，Attention未准入，允许最后一次QKV/O拆分

固定完整logits门Max≤0.1/RMS≤0.02且token一致。FFN-only为5.153/1.294并改token；Attention-only
为0.161/0.0346、token一致、554.1 tok/s，但仍超门。两个scope都不默认；FFN关闭，Attention
只允许再拆QKV/O一次。

回归：CPU 421/421、ASan/UBSan 418/418；PyTorch-enabled CPU注册422项，MI300X HIP保持
211项并通过scope相关operator/model/CLI门；RCCL保持55/55。
