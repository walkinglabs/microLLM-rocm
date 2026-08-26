# Qwen3-0.6B官方strict alignment

日期：2026-08-26
状态：官方FP32 smoke aligned

strict streaming有界验证311.2MB tied alias后加载310参数。MI300X与Transformers完整151,936
logits Max/RMS 3.86e-5/8.44e-6，argmax与四token完全一致；load3.154s、常驻2.384GB、短decode
296 tok/s。只声明固定Qwen3-0.6B FP32 smoke，不推广到其他规模/精度/训练。

回归：CPU 429/429、ASan/UBSan 426/426、PyTorch-enabled CPU 431/431、MI300X HIP
213/213、RCCL 55/55；机器审计157个测试文件。
