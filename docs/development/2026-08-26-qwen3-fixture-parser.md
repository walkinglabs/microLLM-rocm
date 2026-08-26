# Qwen3-0.6B fixture与parser

日期：2026-08-26
状态：fixture/parser ready，strict tied alias待实现

固定官方revision、Apache-2.0、config/tokenizer和完整BF16 header。运行时唯一参数596,049,920；
文件存储751,632,384值/311 Tensor，因为embedding与lm_head各311.2MB且逐字节相同。parser、参数
shape和310项mapping通过；不得在strict alias验证和完整logits前声明官方推理兼容。

回归：CPU 426/426、ASan/UBSan 423/423、PyTorch-enabled CPU 428/428；MI300X HIP保持
212/212且核心QK-Norm门已通过；RCCL保持55/55。
