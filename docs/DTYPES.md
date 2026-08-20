# Tensor 数据类型：数字用几位保存，计算时用几位

## 1. 为什么不能只有 FP32

一个 FP32 数字占 4 字节，FP16/BF16 占 2 字节，FP8/INT8 占 1 字节。模型有
一亿个参数时，仅参数本身大约分别需要 400 MB、200 MB 或 100 MB。数字更短
通常能减少显存和搬运时间，但也更容易丢失精度。

因此，“Tensor 支持某个 dtype”至少包含四层：

1. Storage 真正按相应字节数分配；
2. Tensor 能创建、转换、切片、转置、复制和保存它；
3. 算子 Kernel 能读取并计算它；
4. 模型知道哪些步骤必须临时使用更高精度。

只在枚举中加入名字不算支持。

## 2. AMD Instinct 代际边界

| GPU | 架构 | 原生重点类型 | microLLM 策略 |
|---|---|---|---|
| MI300X / MI325X | CDNA3, `gfx942` | FP32、TF32、FP16、BF16、FP8、INT8 | BF16 训练优先；FP8 GEMM；FP4 做软件解包路径 |
| MI350X / MI355X | CDNA4, `gfx950` | 上述类型加 MXFP8/MXFP6/MXFP4 | 增加原生 MX 格式候选 |

MI300X 没有 CDNA4 的原生 MXFP4 Matrix Core。仓库可以保存 packed FP4 权重，
并在计算前解包或反量化到 FP8/BF16/FP16，但必须把它标为 weight-only 软件路径，
不能写成 MI300X 原生 FP4。

事实来源：[MI300X 官方规格](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)、
[AMD CDNA 架构与 CDNA4 MXFP4/MXFP6](https://www.amd.com/en/technologies/cdna.html)、
[ROCm MI300/MI350 优化说明](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/inference-optimization/workload.html)。

## 3. 当前实现状态

| 能力 | FP32 | FP16 | BF16 | FP8 | INT8 | FP4/INT4 |
|---|---:|---:|---:|---:|---:|---:|
| 真正的 Tensor 存储 | ✓ | ✓ | ✓ | MI300 FNUZ ✓ | 计划中 | 计划中 |
| CPU 构造/读取/cast | ✓ | ✓ | ✓ | 计划中 | 计划中 | 计划中 |
| view/contiguous/设备复制 | ✓ | ✓ | ✓ | 计划中 | 计划中 | 计划中 |
| 基础逐元素/SiLU/SwiGLU/GEMM | ✓ | CPU/MI300X ✓ | CPU/MI300X ✓ | — | — | — |
| hipBLASLt GEMM | FP32 ✓ | MI300X ✓ | MI300X ✓ | MI300X E4M3/E5M2 FNUZ ✓ | 计划中 | 软件解包后计算 |
| Transformer Linear 训练/推理 | FP32 | 计划中 | 单份 FFN+Attention 推理 ✓；FP32-master Linear训练 ✓ | FP8 forward + FP32 master/backward/KV decode ✓ | — | — |

表格中的“计划中”不是支持声明。只有对应测试和真机记录完成后才会改成 ✓。

MI300X 的固定 `512³` 实测中，hipBLASLt 相对可读 Kernel 的平均 Event 加速为
FP32 4.47x、FP16 3.83x、BF16 5.60x。原始记录和边界说明见
`docs/development/2026-08-19-mi300x-precision-capabilities.md`。

## 4. 类型提升规则

第一版不做隐式混合，避免程序悄悄改变精度：

```text
add(FP16, FP16)   -> FP16
add(BF16, BF16)   -> BF16
add(FP16, BF16)   -> 报错，调用者必须先 cast
softmax(BF16)     -> 内部 FP32 reduction，输出 BF16
cross_entropy(*)  -> FP32 scalar loss
```

TF32 是 FP32 GEMM 的计算模式，不是 Storage dtype。FP8/INT8/FP4 还必须携带
scale、group/block 大小和 packed layout，不能只有一个 `DType` 枚举。

## 5. PyTorch 怎样作为精度参考

仓库当前主要使用 Python `torch`，不是让核心引擎依赖 LibTorch：

```text
C++ microLLM 运行并输出 Tensor
              ↓ 相同权重、输入、shape、dtype
Python torch 重建算子或模型
              ↓
比较 forward、loss、每个参数 gradient、非有限值和误差位置
```

LibTorch 只用于可选的 Custom Op 桥接层。FP16/BF16 使用
`torch.testing.assert_close` 和 dtype 对应容差；FP8/量化路径还要比较解量化后的
FP32 数值、logits 和端到端生成结果。

## 6. 什么是“激活岛”

可以把 FP32 和 BF16 想成两种宽度不同的作业本。以前每做一道小题，都把答案从宽本
抄到窄本，算完又抄回宽本：

```text
FP32 → BF16 gate → FP32
FP32 → BF16 up   → FP32
FP32 SwiGLU → BF16 down → FP32
```

抄写本身要花时间。连续 BF16 FFN 把三道相邻题放在同一本窄本里：

```text
FP32 → BF16 gate/up → BF16 SwiGLU → BF16 down → FP32
         只进入一次                         只离开一次
```

矩阵乘法仍用 FP32 累加，减少的是中间结果占用和重复转换。`bf16_ffn` 要求输入是连续
二维 FP32，三个权重是同一设备上的连续二维 BF16；shape 不匹配会直接报错。输出回到
FP32，方便与 residual 相加。

官方 Qwen/DeepSeek 现在可以把 FFN 与 Q/K/V/O projection 权重单向准备为单份 BF16，
Q/K/V 共享一次 input cast；Embedding、Norm、KV cache 与 tied 输出头仍是 FP32。因此
准确名称是“BF16 Linear 混合推理”，不是“整网 BF16”。早期 DeepSeek decode 低于
PyTorch 全 BF16 的问题已由 BF16 专用 immutable hipBLASLt plan 修复：固定短 prompt
Qwen/DeepSeek 四项 inference throughput 全部过线。这个结论仍不能推广到训练、长上下文、
batch>1、Radeon 或其他 ROCm 版本。

训练时不能删除 FP32 master。`LinearPrecision::BFloat16` 只让 Linear forward 使用 BF16
舍入，backward、参数和 AdamW 仍为 FP32。官方多步 loss 与 PyTorch BF16 autocast 接近，
但当前比 microLLM FP32 慢约 8%–9%，峰值不降；continuous BF16 training island 仍未完成。
