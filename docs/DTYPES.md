# Tensor 数据类型：数字用几位保存，计算时用几位

> 当前新增的 BF16 weight-gradient 原语会把两个 FP32 操作数显式舍入为 BF16，
> 再用 FP32 累加和输出。CPU、HIP 与 PyTorch oracle 对齐的是这套 BF16 数学，
> 不是 FP32 gradient 的 bit-exact 结果。20-step模型门失败后，gate/up-only Autograd
> 路由已删除；query/KV 的实测反例也禁止把它扩展成全局策略。

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
| hipBLASLt GEMM | FP32 ✓ | MI300X ✓ | MI300X ✓ | MI300X E4M3/E5M2 FNUZ ✓ | raw INT8×INT8→INT32 probe ✓；公共API未实现 | 软件解包后计算 |
| Transformer Linear 训练/推理 | FP32 | 计划中 | 单份 FFN+Attention 推理 ✓；FP32-master Linear训练 ✓ | 单份Linear权重与per-Tensor weight amax ✓；official精度仍失败，不是可用模型策略 | — | — |

表格中的“计划中”不是支持声明。只有对应测试和真机记录完成后才会改成 ✓。

MI300X 的固定 `512³` 实测中，hipBLASLt 相对可读 Kernel 的平均 Event 加速为
FP32 4.47x、FP16 3.83x、BF16 5.60x。原始记录和边界说明见
`docs/development/2026-08-19-mi300x-precision-capabilities.md`。

后续 128/256/512/1024 square GEMM roofline 表明，FP8 并非自动最快：128–512 相对
hipBLASLt FP32 为 `0.919×–0.966×`，1024³ 才达到 `1.107×`；该点 FP8 为13.62 TFLOPS，
只占 MI300X 2614.9 TFLOPS 官方峰值的0.52%。当前矩阵最快是FP16的18.63 TFLOPS。
这证明部分模型decode超过PyTorch来自更短的软件路径，不是GEMM已经吃满硬件。详见
[Experiment 119](optimization-log/experiments/119-mi300-precision-roofline.md)。

2048/4096 使用显式 FP32 GPU reference 后，FP8 分别达到99.06/477.19 TFLOPS，是FP32的
1.73×/4.31×、FP16的1.10×/1.42×。4096 FP8仍只占官方峰值18.25%，而FP32达到67.83%。
大矩阵证明FP8有真实加速，也证明当前低精度路径尚未饱和。详见
[Experiment 120](optimization-log/experiments/120-large-precision-roofline.md)。

独立raw INT8 probe已实际提交hipBLASLt Kernel：4096³达到416.03 TOPS、官方峰值15.91%，每个
shape五个CPU整数抽样点exact。它不改变上表中Tensor/Transformer INT8仍未实现的状态；详见
[Experiment 121](optimization-log/experiments/121-int8-executed-probe.md)。

官方Qwen/DeepSeek静态scale FP8矩阵已能完整执行并显著降低resident weights，但T8/T512四个
aggregate全部超过`max≤0.2/RMS≤0.05`门，Qwen T512还改变top token。Deep T8包含1个不受
native FP8支持的down shape和112次BF16软件回退。准确状态是“模型执行地基已建立，scale策略
不可用”，详见[Experiment 122](optimization-log/experiments/122-official-fp8-static-scale.md)。

固定4×4网格和连续三段边界实验共证明：不同模型需要冲突的activation scale，有限枚举不能成为
跨模型策略。per-Tensor weight amax把四个官方RMS相对最初静态点降低39%–78%，但仍全部超过
完整logits门；一次性准备约2.8秒/12.2秒。准确状态仍是opt-in研究路径，不是可用模型默认。
详见[Experiment 123](optimization-log/experiments/123-fp8-global-scale-grid.md)至
[Experiment 127](optimization-log/experiments/127-fp8-tensor-amax-weight.md)。

device per-input-Tensor activation amax不经host传递scale，把四个RMS再降低63%–81%，但仍为门的
3.85×–8.76×；single-block reduction还让T512相对BF16只剩4.4%–5.3%吞吐。device-scale
基础设施已实现，当前模型策略和Kernel都未接受。详见
[Experiment 129](optimization-log/experiments/129-fp8-device-activation-amax.md)。

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
Q/K/V 共享一次 input cast；Embedding、Norm 与 tied 输出头仍是 FP32。KV Cache默认
FP32，也可显式选择BF16 Storage并保持FP32 Attention累加。由于DeepSeek T512 B1的
Release RMSE超过固定门，且普通构建出现过build-sensitive token分叉，它还不是默认策略。
因此准确名称仍是“BF16 Linear混合推理”，
不是“整网BF16”。早期 DeepSeek decode 低于
PyTorch 全 BF16 的问题已由 BF16 专用 immutable hipBLASLt plan 修复：固定短 prompt
Qwen/DeepSeek 四项 inference throughput 全部过线。这个结论仍不能推广到训练、长上下文、
batch>1、Radeon 或其他 ROCm 版本。

诊断时可以调用`prepare_bf16_ffn_inference(fp32_layers)`，或在HF CLI传
`--bf16-ffn-fp32-layers`，让少数Block的FFN留在FP32。它不是同时保存两份权重：被选Block只留
FP32，其余Block只留BF16。这样可以用实际converted count和完整logits回答“误差主要从哪几层
注入”，而不是靠最终token猜测。层索引错误、重复或没有启用BF16 FFN都会立即失败。

KV Cache的形状、字节公式、API和精度失败见
[KV Cache数据类型设计](dev/kv-cache-dtypes.zh-CN.md)。

逐层Cache策略允许敏感层保留FP32。固定DeepSeek实验中，layer 1只对原prompt通过；
layers 0–3为FP32的robust-strict在四类prompt上14/14通过，Cache仍比全FP32小1.75×。
它是显式策略，不是模型名触发的隐式默认。

Qwen的constant T2048反例无法由前4/8/12层FP32修复，只有全FP32 Cache通过。低精度KV
不能被描述为对所有prompt同步精度；调用方必须保留FP32 fallback。

训练时不能删除 FP32 master。`LinearPrecision::BFloat16` 只让 Linear forward 使用 BF16
舍入，backward、参数和 AdamW 仍为 FP32。官方多步 loss 与 PyTorch BF16 autocast 接近，
但当前比 microLLM FP32 慢约 8%–9%，峰值不降；continuous BF16 training island 仍未完成。
