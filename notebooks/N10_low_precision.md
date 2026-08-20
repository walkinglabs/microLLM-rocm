# N10 — FP16、BF16、FP8：数字变短后真的更快了吗

本章先问一个很朴素的问题：同一个数字少用几个二进制位保存，会发生什么？

```text
FP32   每个数字 4 字节，范围和精度都较宽
FP16   每个数字 2 字节，小数较细但范围较小
BF16   每个数字 2 字节，范围接近 FP32、小数较粗
FP8    每个数字 1 字节，必须配合 scale 使用
```

数字更短可能减少显存和搬运，但“占用更小”“硬件支持”“Kernel 更快”“整网更快”
是四个不同结论，必须分别证明。

## 运行前先预测

给定：

```text
x = [0.01, 0.1, 1, 10, 1000]
```

请预测：哪个格式最容易保留 `0.01` 的细小变化？哪个格式最不容易让 `1000`
溢出？不要先运行程序。

## “支持 dtype”至少要过四层

```text
Storage 真的按 1/2/4 字节分配
  ↓
Tensor 可以创建、view、copy、cast 和读取
  ↓
Kernel 真正用该格式计算
  ↓
模型知道哪些位置必须保留更高精度
```

只在枚举中加入 `FP8` 名字不算支持。把 FP32 数组贴上 FP8 标签也不算。

## FP16 和 BF16 怎样选

- FP16 小数位更多，数值范围比 BF16 小；
- BF16 范围接近 FP32，训练中通常更不容易溢出；
- MI300X 两者都有矩阵计算硬件；
- 最终选择仍要看真实 shape、误差和端到端时间。

Softmax、RMSNorm reduction、loss 和优化器状态通常需要临时升到 FP32，不能因为
输入是 BF16 就把每一步都强制留在 BF16。

## FP8 为什么必须带 scale

本框架采用明确约定：

```text
q = round(x / scale)
恢复值 = q × scale
```

`ScaledTensor` 同时保存 FP8 数据和 FP32 scale。如果忘记 scale，同一串 FP8 位
无法恢复原来的量级。

MI300X 的 FNUZ 格式中：

| 格式 | 特点 | 常见用途 |
|---|---|---|
| E4M3FNUZ | 小数更细，范围较小 | 权重和前向激活 |
| E5M2FNUZ | 范围更大，小数更粗 | 动态范围大的梯度或激活 |

实用选择方法：

```text
先尝试 E4M3
  ↓
测 amax、饱和比例、变零比例和反量化误差
  ↓
范围过大时比较 E5M2
  ↓
两者都不能过 logits/loss/gradient 门时退回 BF16 或 FP32
```

当前框架底层支持 E4M3FNUZ 和 E5M2FNUZ 的存储、转换与 scaled GEMM；模型 Linear
策略当前使用 E4M3FNUZ。动态 amax history 和自动 E4/E5 选择仍未完成。

## 运行 MI300X 精度 Benchmark

```bash
"$MICROLLM_ENGINE_DIR/build/hip-release/benchmarks/microllm_bench_precision" \
  --size 512 \
  --warmup 3 \
  --repetitions 10
```

同一程序依次运行：

```text
readable FP32
hipBLASLt FP32
hipBLASLt FP16
hipBLASLt BF16
hipBLASLt FP8 E4M3FNUZ
```

每条结果都包含 median、p95、最大绝对误差和两个 speedup 基线。速度结果没有通过
误差门时，程序返回失败；不能先挑最快结果再忽略数值。

## 已有 512³ 记录怎样读

2026-08-19、MI300X、输入预先转换后的记录：

| 路径 | median | 对 hipBLASLt FP32 | 最大绝对误差 |
|---|---:|---:|---:|
| readable FP32 | 0.22750 ms | 0.199× | 6.56e-7 |
| hipBLASLt FP32 | 0.04534 ms | 1.000× | 1.43e-6 |
| FP16 | 0.04156 ms | 1.091× | 5.37e-4 |
| BF16 | 0.04943 ms | 0.917× | 6.54e-3 |
| FP8 E4M3 | 0.04025 ms | 1.126× | 5.55e-2 |

可以得出的结论：在这个固定小 shape 上，预量化 FP8 GEMM 比同程序 hipBLASLt
FP32 快约 `1.126×`。

不能得出的结论：

- 不能说所有 FP8 shape 都快；
- 不能说比 PyTorch 整网快 `5.65×`；
- 不能忽略量化、反量化、scale 计算和 Kernel 启动；
- 不能把单个 GEMM 结果写成 Qwen 端到端速度；
- 不能把 MI300X 的 FP8 结果推广到所有 Radeon。

`5.65×` 只是 FP8 与教学用 readable FP32 Kernel 的差距，不是生产框架基线。

## 模型怎样使用 FP8

当前小型 Transformer 使用：

```text
FP32 master weights
  ↓ 按显式 scale 量化
FP8 Q/K/V/O、FFN 和 LM head GEMM
  ↓
FP32 Norm、Softmax、loss、gradient
  ↓
FP32 AdamW 更新 master weights
```

这条路径已完成小模型 forward、backward、更新和 KV Cache decode。官方
Qwen2.5-0.5B 与 DeepSeek Distill 1.5B 当前完成的是 FP32 整网对齐，还没有可信的
BF16/FP8 整网速度比。因此现在只能说“框架具有低精度模型能力”，不能说“官方
Qwen 已经低精度加速多少倍”。

## PyTorch 精度对照

核心引擎不依赖 LibTorch。对照过程是：

```text
microLLM C++ 输出 Tensor/gradient/logits
                ↓ 相同输入、权重、shape、dtype
Python PyTorch 重建计算
                ↓
比较完整数值、非有限值、误差位置和最终 token
```

LibTorch 只属于可选 Custom Op 桥接层。低精度测试必须比较反量化 FP32 值，不能
直接把不同 FP8 位模式当作普通整数比较。

## 必做反例

固定一个 shape 和输入，只改变 scale：

1. scale 太小，让大值饱和；
2. scale 太大，让小值大量变成 0；
3. 分别记录饱和比例、零比例、最大误差和 GEMM 输出误差；
4. 说明哪一个指标最先推翻“这个 scale 合适”。

进阶任务是设计动态 scale，但一次只改变一种策略：per-tensor、per-channel 或
block scale，不能同时改变格式、分块和模型。

## 本章提交物

- dtype 字节数和 MI300X 能力报告；
- E4M3/E5M2 选择理由；
- 固定 shape 的原始 Benchmark 输出；
- FP32/FP16/BF16/FP8 数值误差；
- 包含量化成本的端到端复测；
- 一个“算子变快但模型没有变快”的反例，或者证明该反例没有出现；
- 明确区分已经支持、仅有接口和尚未验证。
