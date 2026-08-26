# INT8权重格式：先把数字装对，再谈计算加速

这份文档用尽量简单的话解释microLLM当前的INT8能力。先说结论：框架已经能把浮点Tensor
压成一字节INT8、带着scale保存到safetensors、在CPU或AMD GPU上还原，并与PyTorch逐值
对齐。Transformer现在有显式、单向的INT8权重准备研究入口，但M>1 prefill会回到反量化基线，
官方模型也尚未验收，因此不能把它叫作“通用INT8整模推理加速”。

## 1. 为什么一个INT8数字还不够

FP32像一把刻度很细的尺子，一个数字占4字节。INT8只有从-128到127的256个整数，一个数字
占1字节。若直接把`0.125`塞进整数，它会丢失小数部分。

解决办法是再保存一把尺子的刻度`scale`：

```text
真实值 ≈ 整数值 × scale
```

例如`scale = 0.25`：

```text
真实值  -1.50  -0.50   0.00   0.75
INT8        -6      -2      0      3
还原值  -1.50  -0.50   0.00   0.75
```

因此，microLLM不用一个裸INT8 Tensor冒充完整权重，而是使用一对Tensor：

```cpp
struct Int8ScaledTensor {
    Tensor values;  // int8，shape与原权重相同
    Tensor scale;   // float32，只有一个数，和values在同一设备
};
```

## 2. 量化规则

当前第一版是“每个Tensor一个scale”的对称量化：

```text
q = clamp(round_to_nearest_even(x / scale), -127, 127)
还原 = q × scale
```

- 使用`-127…127`，故意不使用`-128`，让正负范围对称；
- `round_to_nearest_even`与PyTorch `torch.round`一致，例如`2.5→2`、`3.5→4`；
- 太大的数饱和到边界，不发生整数回绕；
- NaN变成0，正负无穷分别饱和到`127/-127`；
- scale必须是有限正数；
- 输入可以是连续FP32、FP16或BF16；还原输出可以选择这三种浮点格式。

## 3. 基本API

```cpp
#include <microllm/ops/ops.h>

auto fp32 = microllm::Tensor::from_vector(
    {-3.0F, -1.5F, 0.0F, 1.5F, 3.0F, 31.75F}, {2, 3});

auto packed = microllm::ops::quantize_int8(fp32, 0.25F);
auto restored = microllm::ops::dequantize_int8(
    packed, microllm::DType::Float32);
```

在HIP上，`values`和`scale`都留在显存。创建固定scale时只上传4字节元数据；随后
`dequantize_int8`的Kernel直接读取设备scale，不先把它抄回CPU。

## 4. safetensors怎样保存

约定使用两个名字：

```text
linear.weight        I8   [output, input]或框架要求的实际shape
linear.weight.scale  F32  []
```

`Preserve`表示混合文件里的I8和F32保持自己的格式：

```cpp
microllm::io::StateDict state{
    {"linear.weight", packed.values},
    {"linear.weight.scale", packed.scale},
};

microllm::io::save_safetensors(
    "linear-int8.safetensors", state,
    {.dtype = microllm::io::WeightFileDType::Preserve});
```

加载后重新组成这一对即可：

```cpp
auto state = microllm::io::load_safetensors(
    "linear-int8.safetensors", microllm::Device::hip(0));

microllm::ops::Int8ScaledTensor packed{
    state.at("linear.weight"),
    state.at("linear.weight.scale"),
};
auto fp32_on_gpu = microllm::ops::dequantize_int8(packed);
```

这里没有隐藏的zero-point。若未来加入逐通道scale、zero-point或INT4，必须用新的明确合同，
不能让同一个文件名在不同程序里产生不同解释。

## 5. 到底省多少内存

若有`N`个权重：

```text
FP32       4 × N 字节
当前INT8   1 × N + 4 字节
```

当N很大时，权重payload接近减少75%。这只是权重表示大小，不等于整机显存减少75%；模型还有
activation、KV Cache、临时workspace，以及目前为计算而还原出的浮点Tensor。

## 6. 怎样证明不是自己骗自己

仓库同时保留四类门：

1. Tensor门：一字节Storage、signed值、transpose/slice view逻辑顺序；
2. CPU/HIP门：FP32/FP16/BF16输入的每个量化字节相同，反量化热路径零payload传输；
3. PyTorch门：独立Python程序用`torch.round + clamp + int8`生成答案；
4. 文件门：C++与官方`safetensors.torch`双向写读I8权重和F32 scale。

聚焦命令：

```bash
ctest --test-dir build/cpu-debug --output-on-failure \
  -R 'Int8|SafetensorsTest.Preserve'

ctest --test-dir build/hip-release --output-on-failure \
  -R 'HipInt8|LoadsMixedInt8'
```

PyTorch oracle只在配置所用的Python能`import torch`时注册。官方safetensors互操作使用
`MICROLLM_SAFETENSORS_PYTHON`指定隔离环境。

## 7. 当前不能声称什么

- 没有把官方Qwen/DeepSeek整模参数转换为这套INT8格式；
- `int8_weight_matmul`已经提供正确性基线，但会先还原完整浮点权重，因此不是INT8加速；
- 没有通过M>1与官方模型门的默认Attention/FFN路线；
- 没有融合“读INT8、乘scale、做GEMM”的Kernel；
- 没有INT8训练、量化感知训练或校准数据流程；
- 没有本节点的端到端tokens/s加速结论。

MI300X原始INT8矩阵硬件探测已经证明硬件能执行高速INT8×INT8→INT32，但“硬件能算”和
“模型已经走这条路径”是两件事。下一节点应先做一个带完整输出PyTorch门的weight-only Linear，
分别测“先反量化再GEMM”和“融合/原生INT8 GEMM”；当前显式Transformer接线正使用这两个
边界，默认仍关闭。

当前正确性基线可以直接调用：

```cpp
auto output = microllm::ops::int8_weight_matmul(input, packed);
```

它只接受`input [M,K]`与`packed.values [K,N]`，输出dtype跟随FP32/FP16/BF16 input。实现等价于：

```text
完整 weight = dequantize_int8(packed, input.dtype)
output = matmul(input, weight)
```

这条路径的价值是固定shape、舍入、错误与完整输出答案。因为临时weight重新变成2或4字节，
不能用一字节文件大小推导它的运行峰值，也不能把它的速度称为INT8硬件速度。

研究者可显式选择`Int8WeightMatmulImplementation::FusedDecode`。它只接受HIP FP32 `[1,K]`，
直接读取I8权重并按输出列归约，不生成浮点weight。Qwen/DeepSeek实测显著快于“每次反量化”，
但DeepSeek仍只有PyTorch常驻FP32 GEMM的0.494×，所以`Auto`没有切换。M>1、FP16/BF16 input继续
使用正确性基线。

HIP整模准备使用`quantize_int8_dynamic`：amax、scale和I8 payload都在GPU产生。官方Qwen准备
已做到168个Linear、1.431GB扫描、权重D2H为0；但完整logits和token严重失败，因此
`--int8-linear true`只是显式研究开关，不能用于声称正确的Qwen推理。
