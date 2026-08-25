# Experiment 231 — 去掉三次cast，为什么模型仍然慢

Status: `reject direct-BF16 model route; close online model track`

## 对上轮解释做反驳实验

Experiment 230认为每层RoPE后Q/K/V三次FP32→BF16 cast可能吞掉online收益。本轮只消除这三次：

- `bf16_qkv_projection_out_`可同时保留BF16 Q/K/V；
- `add_bias_bf16`把BF16 V加FP32 bias后直接写BF16；
- `rope_split_half_bias_bthd_bf16`把bias、RoPE和BTHD→BHTD合成一次BF16输出；
- online Attention直接消费三者。

两个新算子都有CPU、HIP、PyTorch与非法shape门。grouped QKV测试验证V fallback与FP32 reference
舍入一致，模型不读取未写入的FP32 output。

## 同一36进程模型门

workload、权重、indices、token、warm-up、measured steps、logits和门全部不变。所有candidate仍精确
命中Qwen 168次、DeepSeek 196次native，零fallback。

| Model/case | 三cast旧候选 | direct BF16 | Peak saved | Direct logit Max/RMS |
|---|---:|---:|---:|---:|
| Qwen B1T256 | 0.783× | 0.824× | 3.8 MiB | 0.185 / 0.041 |
| Qwen B1T1024 | 0.761× | 0.777× | 57.0 MiB | 0.485 / 0.110 |
| Qwen B2T512 | 0.865× | 0.888× | 29.0 MiB | 0.274 / 0.055 |
| DeepSeek B1T256 | 0.843× | 0.867× | 3.5 MiB | 0.105 / 0.019 |
| DeepSeek B1T1024 | 0.763× | 0.781× | 50.0 MiB | 0.094 / 0.016 |
| DeepSeek B2T512 | 0.884× | 0.906× | 26.0 MiB | 0.096 / 0.016 |

![Direct BF16 model discard](../assets/rocwmma-direct-bf16-model-discard.svg)

六格都略有恢复，证明cast确实有成本；但没有一格接近1.0，更没有通过1.01。Qwen完整logits仍失败。
因此“cast是主要瓶颈”被实验推翻，剩余差距来自online kernel在逐层模型中的调度/执行效率与BF16
probability误差本身。

## 决定

- 保留`add_bias_bf16`、direct-BF16 RoPE和QKV value retention原语及测试；
- 模型默认和CLI推荐不变；实验开关仍只用于复现；
- online Attention模型track关闭，不再扫tile、threads或context；
- 未来若重开，必须改变更大的算法/调度边界，而不是继续删除局部cast。

原始证据位于
[`benchmarks/results/2026-08-25-rocwmma-direct-bf16-model-gate/`](../../../benchmarks/results/2026-08-25-rocwmma-direct-bf16-model-gate/)。

发布回归为CPU 341/341、ASan/UBSan 339/339、PyTorch-enabled CPU 315/315、完整CPU/HIP
537/537（3个条件跳过）、HIP标签184/184、RCCL标签14/14与multi-GPU 12/12；覆盖清单注册
103个测试文件。
