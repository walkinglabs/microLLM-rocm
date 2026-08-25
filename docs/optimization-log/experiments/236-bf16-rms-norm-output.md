# Experiment 236 — 不再先写FP32，RMSNorm直接交给BF16投影

Status: `keep operator; model route still closed`

## 为什么这不是降低公式精度

当前BF16 FFN边界执行：

```text
FP32 RMSNorm output
→ 完整写全部FP32元素
→ 再读一次并cast到BF16 Arena
```

新`rms_norm_bf16_out_`仍用FP32求和、FP32 inverse RMS和FP32 weight乘法，只在最后一次
store时舍入BF16。所以它应与“旧GPU RMSNorm + GPU cast”逐位bit-identical。

## 先修正reference

第一版HIP测试拿CPU reduction的BF16结果要求位级相同。由于CPU/GPU block reduction顺序不同，
少数元素差1个BF16台阶，测试正确失败。正式reference应是当前GPU FP32 RMSNorm后在
GPU cast，因为候选要替换的就是这条路径。修正后全shape位级相同。

## 正式operator gate

baseline/candidate都写caller Storage，每shape 3个fresh processes、3 warm-up、30 Event：

| Shape | Event speedup | Wall speedup | Complete output | Timed transfers |
|---|---:|---:|---|---:|
| Qwen B1T1024 D896 | 1.866× | 1.399× | bit-identical | 0 |
| DeepSeek B1T1024 D1536 | 2.070× | 1.511× | bit-identical | 0 |

![Direct BF16 RMSNorm output](../assets/bf16-rms-norm-output.svg)

## 决定

- 保留FP32 caller-output `rms_norm_out_`和直接BF16 `rms_norm_bf16_out_`；
- CPU保留容易检查的reference，HIP路径零payload transfer；
- 这一节只准入算子，不改Transformer默认；
- 下一节才把FFN Norm连到现有Arena，并重新跑完整logits、吞吐和显存门。

发布回归：CPU 345/345、ASan/UBSan 343/343、PyTorch-enabled 319/319、完整CPU/HIP
544/544（3个条件跳过）、HIP标签187/187；覆盖清单注册107个测试文件。

证据：[`operator matrix`](../../../benchmarks/results/2026-08-25-bf16-rms-norm-output-operator/)
