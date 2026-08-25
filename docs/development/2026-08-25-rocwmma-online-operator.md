# 2026-08-25 — 快速实验怎样变成别人能调用的算子

## 私有实验和公共积木的区别

私有benchmark知道自己永远只收T32的倍数，也知道只有一张卡。公共算子不能让用户记这些秘密。
它要检查dtype、shape、batch、head关系、设备和连续性；能走快路时走快路，不能时仍给出正确结果。

新算子只接受BF16 Q/K/V，永远返回FP32。MI300X gfx942、T整32、D64/128走online矩阵单元；
T31、T33、D32或其他设备先转FP32，再走旧实现。这个fallback较慢，但比静默算错或直接崩溃好。

## 怎样证明没有走错路

算子有两个计数器：native调用数和fallback调用数。正式测试每个进程先清零：

- 原生case做5次热身和20次测量，必须得到25/0；
- fallback case必须得到0/25；
- 两者都比较全部输出，而不是只看计数器。

一次T33调用如果意外进了矩阵kernel，整除合同会被破坏；一次T32调用如果总fallback，速度数据也
没有意义。计数与数值必须同时通过。

![Public online operator](../optimization-log/assets/rocwmma-online-operator.svg)

公共原生路径在10个B1/B2 case中比当前算子快1.534–2.456倍。fallback只有0.607–0.696倍，
但数值几乎精确。现在可以进入模型实验，还不能默认启用，因为一次模型有很多层，局部约5e-4
误差可能累积到logits。

## 发布回归

CPU 340/340、ASan/UBSan 338/338、PyTorch-enabled CPU 314/314、完整CPU/HIP 536/536
（3个条件跳过）、HIP标签184/184、RCCL标签14/14、multi-GPU 12/12，覆盖清单注册102个
测试文件。CPU/HIP的build-tree与搬迁
install-tree Config以及公开example均为3/3；CMake feature metadata没有扩大外部链接依赖。
