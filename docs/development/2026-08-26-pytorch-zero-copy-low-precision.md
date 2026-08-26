# FP16/BF16也不搬家

日期：2026-08-26
状态：PyTorch ROCm multiply/matmul caller-owned输出已验收

## 改动很小，证据必须完整

已有Tensor和HIP Kernel早已支持FP16/BF16。本节点没有复制实现，只给C ABI `ml_dtype`和Python
`DType`补上明确枚举，并让external描述符映射到现有`DType::Float16/BFloat16`。

CPU测试直接用IEEE FP16和BF16的16位编码包装外部buffer，运行caller-owned multiply并读回3和8。
这能发现枚举错位、字节数错误和Python `tolist()`把低精度误当Int32等问题；实现中确实先被该测试
发现并修复了一次`tolist`分支。

## 真机门

每个dtype使用真实PyTorch ROCm指针：

```text
4,194,304元素 multiply_out ×64
1024×1024 matmul_out ×64
Torch Event记录/等待
比较完整PyTorch Tensor的Max
```

三轮共6个dtype-run组合，36个外部指针逐个一致且不拥有。12/12 Event在query时pending，multiply和
matmul Max均为0。每轮60MiB，合计180MiB，wrapper copy 0字节，共768次提交。

![Low precision zero-copy](../../benchmarks/results/2026-08-26-pytorch-zero-copy-low-precision/zero-copy-low-precision.svg)

## 失败如何改变实验

8次BF16 matmul的pilot偶尔在query前已完成。它推翻的是“短工作一定能看到pending”，不是算子正确性。
正式门把同一个预分配工作重复64次；没有加入sleep或同步伪造异步状态。

## 下一边界

外部FP16/BF16只覆盖multiply/matmul。add的low-level TensorView仍限定FP32；Softmax、RMSNorm、
Attention和训练梯度尚无完整caller-owned C/Python面。随机shape误差矩阵与混合进程profiler也仍需做。
