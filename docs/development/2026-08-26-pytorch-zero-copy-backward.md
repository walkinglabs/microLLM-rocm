# Backward梯度写进调用者的Tensor

日期：2026-08-26
状态：六个基础backward家族已验收

## 新接口

- Softmax input gradient out；
- RMSNorm input/weight gradient + row inverse-RMS workspace；
- SwiGLU gate/up gradient；
- RoPE input gradient；
- CrossEntropy logits gradient + row stats/factor workspace；
- Embedding weight gradient add。

所有当前backward接口要求FP32梯度。可写Tensor必须互不alias输入/彼此、同shape/device且contiguous。
CPU参考可能创建临时结果用于教学对照；HIP路径直接写caller payload，不创建返回Tensor。

## 三seed PyTorch autograd矩阵

114条完整输出、10组梯度目标。285/285指针/ownership通过，wrapper copy 0。最大Max/RMS为
`8.59e-6/1.42e-6`，来自64×1024 RMSNorm weight reduction；它只占门的10.7%。SwiGLU双梯度Max
`9.54e-7`，RoPE`4.77e-7`，CrossEntropy logits`2.99e-8`，Embedding重复index为0误差。

![Backward matrix](../../benchmarks/results/2026-08-26-pytorch-zero-copy-backward/backward-matrix.svg)

## 语义边界

Embedding backward是累加而非覆盖；测试先清零外部weight gradient。RMSNorm/CrossEntropy scratch是
公共workspace但其payload不属于稳定输出API，只有相应明确字段（inverse-RMS/factor）进入对照。

下一步是让Autograd叶子梯度直接绑定外部池，并跑一次完整Transformer forward/backward；不能把114条
孤立算子对照写成模型训练零复制。
