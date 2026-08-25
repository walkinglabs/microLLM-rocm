# 2026-08-25 — 不先写一张马上要丢掉的表

旧程序先把RMSNorm的结果全部写成FP32，然后马上又读回来改成BF16。新算子前面的
计算都不变，只在最后写结果时直接写BF16。

两个真实shape的完整数组与旧GPU路径每一位都一样，设备时间快1.87倍和2.07倍。

![Direct BF16 RMSNorm output](../optimization-log/assets/bf16-rms-norm-output.svg)

这一步只是准备好一个零件。它还没接到模型默认路线；下一步会另外测整个模型。

最终回归为CPU 345/345、消毒343/343、PyTorch-enabled 319/319、完整CPU/HIP
544/544和HIP 187/187；只有3个既有条件跳过。
