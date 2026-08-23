# Matmul注册键：不能只记住三个数字

## 初中生版解释

旧注册表像一排只写“64×64×64”的抽屉。我们在FP32测试后把“用hipBLASLt”纸条放进去，
下一次FP16、转置矩阵、另一张GPU甚至不同ROCm版本也会摸到同一张纸条。shape相同不代表题目
相同，这种优化可能数值正确，也可能变慢或直接不支持。

新`MatmulTuningKey`像完整快递地址，包含：

```text
M/K/N
dtype
左/右是否转置
左右stride
GPU architecture
HIP runtime / driver / hipBLASLt version
inference / training / unspecified mode
workspace上限
```

Matmul本身就是op身份，因此键中不再重复字符串`matmul`。以后扩展到其他op时，每种op拥有
自己的强类型键，不能把Softmax候选塞进Matmul注册表。

## API

```cpp
microllm::ops::OpContext context;
context.mode = microllm::ops::OpMode::Inference;
context.workspace_bytes = 4 * 1024 * 1024;

const auto key = microllm::ops::make_matmul_tuning_key(
    left, right, false, true, context);
microllm::ops::register_matmul_implementation(
    key, microllm::ops::MatmulImplementation::HipBLASLt);
```

调用者从真实Tensor生成key，不手写GPU名或stride。注册表是进程内、mutex保护的；空注册表热路径
只做一次atomic读取，不构造string/vector，也不查询设备属性。

## 反例测试

gfx942测试先只注册FP32 NN 64³，然后证明以下选择仍保持Readable：

- 同shape FP16；
- 同shape TT；
- Training mode；
- 4KiB workspace，而注册项是0 workspace。

CPU测试检查host/version/stride/mode/workspace字段，并拒绝空architecture、Auto实现和非连续
Tensor。清空后entry count必须回到0。

原始targeted test日志与机器摘要保存在
[`benchmarks/results/2026-08-23-matmul-exact-key/`](../../benchmarks/results/2026-08-23-matmul-exact-key/)。

## 仍然没做完

这次修的是“候选不能串门”，不是自动调优器。当前仍缺：

- correctness-before-timing候选执行器；
- Event热身、重复、中位数/P95；
- 可持久化JSON cache及版本失效；
- Attention/Softmax/RMSNorm等其他热点注册表；
- 自动把模型训练/推理语义传进所有调用点。

在这些门完成前，注册API只能接收已经由外部实验验证的选择，不能看到一次最快时间就自动写入。

## 最终回归

Exact-key反例通过后，完整回归还抓到一个与registry无关的旧Attention reduction竞争；旧revision
同样可以复现。修复过程单独记录在
[Experiment 156](../optimization-log/experiments/156-block-reduction-determinism.md)。最终门为：

```text
CPU Debug                 252/252
ASan/UBSan                250/250
PyTorch-enabled CPU       226/226
完整CPU/HIP               370/370（2个条件跳过）
HIP标签                   114/114
```
