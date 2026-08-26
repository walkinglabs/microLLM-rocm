# Qwen3-0.6B显式BF16推理

日期：2026-08-26
状态：固定模型显式策略通过

196个Linear BF16、Q/K-Norm FP32。以FP32共同oracle比较，microLLM Max/RMS 0.0724/0.0142，
Transformers BF16 0.188/0.0371；四token一致。2+5端到端216.6 vs59.2 tok/s，常驻2.384→
1.503GB。只保留固定Qwen3显式策略，不改默认。

回归：CPU 431/431、ASan/UBSan 428/428、PyTorch-enabled CPU 433/433、MI300X HIP
214/214、RCCL 55/55；机器审计158个测试文件。
