# Experiment 135：少做43%量化，Deep T512首次快过BF16

Q/K/V共享一次量化，gate/up共享一次。每次forward调用：Qwen168→96，Deep197→113；三次measured
加一次warmup后分别384/452，机器计数完全匹配。

| 模型 | Exp133 TPS | shared TPS | 提升 | shared/BF16 | RMS |
|---|---:|---:|---:|---:|---:|
| Qwen T512 | 75518 | 85193 | **12.81%** | 0.923× | 0.29251 |
| Deep T512 | 44975 | 50546 | **12.39%** | **1.028×** | 0.24914 |

![Shared activation quantization](../assets/fp8-shared-activation-quantization.svg)

两个max/RMS与Exp133逐值相同，top token全相同。Deep FP8热路径首次略快于BF16，但完整RMS仍
是0.05门的4.98倍，Qwen为5.85倍；速度不能把精度红门改绿。

共享量化作为无损性能优化保留。下一热点不能再从调用重复猜测，应重跑profile确认dynamic三段与
GEMM的新比例；数值路线则需要校准/更细粒度，仍与性能路线分开。
