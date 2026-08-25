# Experiment 230 — 算子快两倍，模型为什么反而慢

Status: `reject model route; retain public operator`

## 模型A/B只改变Attention core

两边都使用同一套BF16 FFN、BF16 Attention权重、FFN/QKV Arena、exact grouped QKV/gate-up、BTHD布局
和retained BF16 Q/K。candidate额外打开一个默认关闭的开关：RoPE后的Q/K/V显式cast BF16，再调用
公共online operator。

```text
current: FP32 Q/K/V → hipBLASLt QK → global probabilities → PV
online:  FP32 Q/K/V → 三次BF16 cast → rocWMMA online QK/PV
```

三次cast是实际模型成本，不能从计时中删除。每个进程由CLI报告native/fallback计数：7次forward
乘24/28层，Qwen必须168次、DeepSeek必须196次；baseline必须0，所有策略fallback必须0。

## 正式矩阵

- 两个固定revision与真实权重；
- B1T256、B1T1024、B2T512；
- current/online，每格3个fresh processes；
- 2 warm-up + 5 measured prefill；
- 每次保存全部B×151936 logits；
- 门：top row相同、Max≤0.2、RMS≤0.02、每格≥1.01×、peak不增加。

## 36进程结果

| Model/case | online/current | Peak saved | Logit Max/RMS |
|---|---:|---:|---:|
| Qwen B1T256 | 0.783× | 3.8 MiB | 0.168 / 0.038 |
| Qwen B1T1024 | 0.761× | 57.0 MiB | 0.511 / 0.112 |
| Qwen B2T512 | 0.865× | 29.0 MiB | 0.314 / 0.061 |
| DeepSeek B1T256 | 0.843× | 3.5 MiB | 0.116 / 0.020 |
| DeepSeek B1T1024 | 0.763× | 50.0 MiB | 0.082 / 0.015 |
| DeepSeek B2T512 | 0.884× | 26.0 MiB | 0.111 / 0.018 |

![Full-model online Attention discard](../assets/rocwmma-online-model-discard.svg)

六格top row仍相同，说明不是完全失控；六格显存也都下降。但所有端到端性能都回退，Qwen三格
还失败预设logit门。算子1.5–2.5倍的收益没有覆盖每层cast与模型其他工作，BF16 probability误差
又经过24层累积。

## 决定

- 默认和CLI行为不变，online模型开关不写入公开推荐；
- public operator、fallback、计数器和operator benchmark保留；
- 不用“top token没变”覆盖完整logits失败；
- 不用57MiB显存收益覆盖24%性能回退；
- 这条模型路线关闭。未来只有在RoPE直接产BF16且避免三次cast，并重新过同一模型门时才能重开。

原始证据位于
[`benchmarks/results/2026-08-25-rocwmma-online-model-gate/`](../../../benchmarks/results/2026-08-25-rocwmma-online-model-gate/)。

发布回归为CPU 341/341、ASan/UBSan 339/339、PyTorch-enabled CPU 315/315、完整CPU/HIP
537/537（3个条件跳过）、HIP标签184/184、RCCL标签14/14、multi-GPU 12/12，覆盖清单注册
103个测试文件。
