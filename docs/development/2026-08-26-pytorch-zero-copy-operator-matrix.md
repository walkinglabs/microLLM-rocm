# 零复制算子不能只测一个漂亮shape

日期：2026-08-26
状态：Softmax/RMSNorm/SwiGLU随机矩阵已验收

## 新caller-owned接口

- `softmax_out_`与C/Python Stream接口；
- FP32 `rms_norm_out`；
- FP32输入/权重、BF16输出的`rms_norm_bf16_out`；
- FP32/FP16/BF16 `swiglu_out`；
- 全部复用现有Kernel并要求外部output不alias输入。

Softmax out只接受FP32 contiguous最后一维。审查实现时发现旧`softmax()`的HIP路径对低精度dtype也会
把data强转成`float*`。现在它在launch前明确拒绝；不能用错误指针解释换取“支持更多dtype”。

## 随机矩阵

三个seed、63条完整PyTorch ROCm输出：Softmax 12、FP32 RMSNorm 12、BF16 RMSNorm output 12、
三dtype SwiGLU 27。宽度覆盖7、64、513、1024，SwiGLU元素覆盖7、4099、65536。

所有指针/ownership为63/63，wrapper copy 0。FP32 Max在`4.47e-8–1.43e-6`；FP16 SwiGLU Max
0.00390625；BF16 SwiGLU Max0.0625、RMS0.001901。BF16 RMSNorm输出与PyTorch BF16结果完全相同。

![Operator matrix](../../benchmarks/results/2026-08-26-pytorch-zero-copy-operator-matrix/operator-matrix.svg)

## 反例改变了什么

原BF16 SwiGLU门0.05在65536元素随机输入上被0.0625推翻。0.0625符合当前量级的一档BF16步长，且
RMS小于0.002，所以门改为0.07并在图中显示已使用89.3%。这不是放弃数值门，而是用可解释的量化
边界替代随意小数。

## 下一边界

Attention、RoPE、Embedding、loss和训练梯度仍缺caller-owned C/Python输出。Softmax FP16/BF16需要
真正typed reduction Kernel，不能通过FP32临时Tensor冒充零复制。
