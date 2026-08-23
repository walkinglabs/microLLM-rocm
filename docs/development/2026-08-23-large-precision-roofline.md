# 2026-08-23：2048/4096大GEMM roofline

## Reference合同

默认benchmark仍用CPU reference。大尺寸必须显式传`--reference fp32`，使用已在小尺寸独立验证的
hipBLASLt FP32输出作为低精度相对基线。它不提供独立大尺寸FP32正确性。

## 结果

- 10/10 accuracy pass，设备门通过；
- 2048 FP8 99.06TF，1.73×FP32、1.10×FP16；
- 4096 FP8 477.19TF，4.31×FP32、1.42×FP16；
- 4096 FP32峰值利用率67.83%，FP8只有18.25%；
- FP8 max error约0.047，输出BF16。

## 决定

保留large reference模式和roofline。FP8真实加速成立，硬件饱和不成立。下一节点只做INT8 executed
probe，不把硬件表中的INT8能力继续当成实现证据。

详见[Experiment 120](../optimization-log/experiments/120-large-precision-roofline.md)。
