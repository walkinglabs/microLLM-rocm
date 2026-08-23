# Experiment 127：每层权重有自己的尺子，仍然不够

## 新设计

静态策略让所有Linear共用weight scale。新`tensor-amax`在准备阶段逐个读取权重：

```text
scale = max(abs(weight)) / 240
FP32 weight → E4M3-FNUZ weight + its own scale
```

Qwen 168个Linear的scale为`0.000350–0.006934`，接近20倍跨度；DeepSeek 197个为
`0.000798–0.007552`，约9.5倍。旧单个0.005确实不可能同时贴合所有层。

## 一次失败先修证据

第一轮完成15条Qwen记录后停止：应用历史字段叫`bf16_prepare_ms`，runner寻找另一个名字，导致
FP8准备时间被写成0。15条精度数据没有冒充正式结果，原样保留且没有summary。

新增通用`weight_preparation_ms`后，3-worker pilot先确认FP8值非零，再从新目录跑正式36条。

## 正式结果

| 模型/T | policy | TPS | prepare ms | resident MiB | max/RMS | top | gate |
|---|---|---:|---:|---:|---:|---|---|
| Qwen T8 | BF16 | 2278 | 15 | 1202 | 0.092/0.0165 | equal | pass |
| Qwen T8 | FP8 amax | 1979 | 2886 | **861** | 4.379/0.667 | equal | fail |
| Qwen T512 | BF16 | 92295 | 15 | 1202 | 0.105/0.0160 | equal | pass |
| Qwen T512 | FP8 amax | 91079 | 2825 | **861** | 6.723/1.298 | equal | fail |
| Deep T8 | BF16 | 1379 | 20 | 4280 | 0.047/0.0083 | equal | pass |
| Deep T8 | FP8 amax | 1377 | 12163 | **2363** | 6.708/1.175 | equal | fail |
| Deep T512 | BF16 | 49463 | 19 | 4280 | 0.045/0.0087 | equal | pass |
| Deep T512 | FP8 amax | 51460 | 12403 | **2363** | 8.480/1.309 | equal | fail |

![FP8 tensor amax weight](../assets/fp8-tensor-amax-weight.svg)

相对Experiment 122的0.025/0.005静态起点，四个FP8 RMS分别下降约77.5%、62.8%、53.8%、
38.9%。这是实质改善，但仍为0.05门的13–26倍，四个gate全部失败。

## 速度、内存和启动成本

FP8/BF16热路径TPS：Qwen T8/T512为0.869/0.987×；DeepSeek为0.999/1.040×。resident内存
保持约861 MiB与2363 MiB。只有Deep T512略快，不能覆盖精度失败。

第一版为可检查性把GPU权重逐Tensor取回host计算amax：Qwen扫描1.431 GB、准备约2.8秒；
DeepSeek扫描6.174 GB、准备约12.2秒。它是一次性成本，不在TPS计时内，但部署冷启动必须报告。

## 决策

保留opt-in API、事务式非有限值门、scale范围报告和零payload-transfer热路径；拒绝把当前策略设为
模型默认。weight尺子已经分开，剩余主要问题是所有层activation仍共用0.2。下一节点先测少量层
activation amax/饱和率，再设计per-row/per-token device scale；在精度过门前不优化2.8/12秒扫描。
