# Experiment 122：FP8省了一半权重，但四个模型精度门全失败

## 前置实现

旧FP8 Linear每次forward重新量化权重。新路径一次性、事务式准备所有Linear：

```text
FP32 Linear weight
→ E4M3-FNUZ one-byte weight + persistent weight scale
→ persistent activation scale
→ FP32 source released
```

Embedding、Norm和tied head仍是FP32。prepare后热路径无scale H2D。

## 第一次正式失败

Qwen18个worker完成，但Deep第一个FP8 worker在`M8×K8960×N1536` down projection返回
hipBLASLt status6。v1立即停止，没有summary。

修复不是假装native支持：exact shape registry记住不支持；该shape把FP8输入/权重反量化到BF16，
再走BF16 GEMM。JSON单独报告native/fallback shape与calls。

## v2正式结果

### Qwen2.5-0.5B

| T | policy | TPS | resident/peak MiB | max/RMS vs FP32 | top | gate |
|---:|---|---:|---:|---:|---|---|
| 8 | FP32 | 1856 | 1885/1886 | 0/0 | equal | pass |
| 8 | BF16 | **2269** | 1202/1203 | 0.092/0.0165 | equal | pass |
| 8 | FP8 | 1602 | **861/862** | **15.10/2.96** | equal | fail |
| 512 | FP32 | 69459 | 1885/1922 | 0/0 | equal | pass |
| 512 | BF16 | **92501** | 1202/1232 | 0.105/0.0160 | equal | pass |
| 512 | FP8 | 91163 | **861/899** | **15.69/3.48** | **9707→23811** | fail |

### DeepSeek Distill Qwen 1.5B

| T | policy | TPS | resident/peak MiB | max/RMS vs FP32 | top | gate |
|---:|---|---:|---:|---:|---|---|
| 8 | FP32 | 1336 | 6779/6781 | 0/0 | equal | pass |
| 8 | BF16 | **1426** | 4280/4281 | 0.047/0.0083 | equal | pass |
| 8 | FP8 | 1377 | **2363/2390** | **17.77/2.54** | equal | fail |
| 512 | FP32 | 28109 | 6779/6847 | 0/0 | equal | pass |
| 512 | BF16 | 49399 | 4280/4326 | 0.045/0.0087 | equal | pass |
| 512 | FP8 | **51558** | **2363/2431** | **11.46/2.14** | equal | fail |

![Official FP8 static scale](../assets/official-fp8-static-scale.svg)

## 速度与内存

FP8 resident：

```text
Qwen     45.7% of FP32, 71.6% of BF16
DeepSeek 34.9% of FP32, 55.2% of BF16
```

FP8/BF16速度：Qwen T8/T512为0.706/0.986×；DeepSeek为0.966/1.044×。只有Deep T512略快，
但它的RMS仍是2.14，不能通过速度门绕过精度。

## Fallback证据

- Qwen T8/T512：4 native shapes，0 fallback；
- Deep T8：4 native、1 fallback、每worker 112 calls；
- Deep T512：5 native、0 fallback，因为M512 shape受支持。

Deep T8吞吐不是纯native FP8；软件回退保留模型可执行性和一字节权重存储，但每次反量化大权重。

## 决策

保留：

- 单份FP8 Linear权重生命周期；
- caller-owned scale零热路径H2D；
- exact-shape native/fallback registry；
- official runner、完整logit和失败数据。

拒绝：

- 固定全局0.025/0.005作为可用模型策略；
- 默认FP8；
- “Qwen/DeepSeek FP8推理已支持”的说法。

下一步只研究scale：至少weight per-tensor、activation per-token/per-row amax，并先在少量层 trace量化
饱和率。不能搜索最终top token后反向挑scale。
