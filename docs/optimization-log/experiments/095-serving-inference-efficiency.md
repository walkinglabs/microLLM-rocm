# Experiment 095 — 推理矩阵加入长短回答、KV浪费和一个不稳定失败

用户真正看到的服务负载不只有短prompt、B1和8个输出token。本节点不改Kernel，先扩大可执行
实验合同，并用空闲MI300X独立跑增量数据。

## 新的固定套件

`--suite serving --cases cached`现在固定为：

```text
context: 1 / 8 / 32 / 128 / 512 / 2048
batch:   1 / 2 / 4 / 8
output:  1 / 8 / 32 / 64
```

每个模型共有96个paired cached shape。它同时包含启动开销占主导的N1、短答N8、中答N32和
长答N64。正式runner仍把warmup单独记录，不计入measured throughput。

显存summary新增五本账：Cache未使用字节、每请求未使用字节、未使用比例、active Cache占增量
峰值、非Cache增量峰值。这样“固定预留了更多页”和“真的泄漏显存”不会混为一谈。

## 增量pilot

先跑两个模型、T1/32/128、B2/4、N64、warmup 1、measured steps 3。24条raw中23条成功，
12个paired case中11个完整配对。

成功case的microLLM/PyTorch吞吐比：

- Qwen：2.92×–3.39×；
- DeepSeek：2.09×–2.44×。

11个成功配对中8个greedy suffix完全一致。Qwen T1 B2/B4在第17个token分叉，DeepSeek T32 B2
在第26个token分叉；这类结果说明只看最终token不够，后续还要在分叉前一步保存logit margin。

## 一个失败为什么不能马上叫稳定bug

pilot中的Qwen T128/B4/N64有一次microLLM batch内部一致性失败：相同输入row没有给出相同token。
原始失败没有删除。但随后用同一冻结Release binary做三个全新进程：

| run | 结果 | tokens/s | peak bytes | KV bytes |
|---:|---|---:|---:|---:|
| 1 | pass | 1132.42 | 1,293,118,976 | 9,437,184 |
| 2 | pass | 1134.66 | 1,293,118,976 | 9,437,184 |
| 3 | pass | 1135.48 | 1,293,118,976 | 9,437,184 |

三次64-token suffix完全相同，吞吐极差约0.27%。所以当前分类是“观察到一次、尚不稳定”，证据
不支持“已有稳定stride bug”。未来若再次出现，runner应保存每个row和首个分叉前的完整logits。

## 真正的长上下文N64

T2048/B2/N64的四条framework raw全部通过，两模型跨框架64 token完全一致：

| 模型 | microLLM tok/s | PyTorch tok/s | micro/PT | micro peak | PT peak | KV（双方） |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 221.28 | 177.04 | 1.250× | 1.777 GiB | 3.433 GiB | 49.5 MiB |
| DeepSeek-Distill-1.5B | 132.46 | 152.63 | 0.868× | 4.871 GiB | 5.943 GiB | 115.5 MiB |

两边active/capacity都是2112 token，KV利用率1.0。每条microLLM记录有384个logical forward，
D2H只有3 calls/1536 bytes，说明64-token history仍按measured step批量复制。

![Serving inference efficiency matrix](../assets/serving-inference-efficiency.svg)

## 结论边界

Qwen长上下文仍领先，但领先从短中context约3×缩到1.25×。DeepSeek短中context领先2.09×–2.44×，
到T2048却只有0.868×，因此长context cached Attention仍是明确热点。

这不是完整96-shape正式发布矩阵：pilot只有新增轴的代表点，每个paired shape只有一个进程。下一步
优先加多进程重复和分叉前logit margin，而不是一次跑完巨大笛卡尔积后才发现失败证据不够定位。

原始结果见 [`095-data`](095-data/)。
