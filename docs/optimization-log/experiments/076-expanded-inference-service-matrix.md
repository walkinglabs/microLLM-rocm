# Experiment 076 — 推理不能只测一个短prompt

> **后续修正：** Experiment 077的rocprof证明，本页prefill行计算并回传了完整`[B,T,V]`
> logits，因此属于`full-logits prefill`，不是严格的服务TTFT。原始数据保留，但服务性能结论
> 已由显式`last-logit prefill`矩阵取代；cached decode与KV结论不受影响。

这次把正式checkpoint推理沿三条轴展开：Qwen2.5-0.5B与DeepSeek-R1-Distill-Qwen-1.5B，
context 32/512/2048，以及batch 1/2/4/8。每个点分开测prefill与有Cache的steady decode；
代表点再比较FP32/BF16 KV Cache。

## 实验合同

```text
GPU          MI300X VF / gfx942
source freeze f087fb4
prefill      48 process rows
cached FP32  48 process rows
cached BF16  24 representative rows（只测B1/B8）
warm-up      1次，不计时
measured     3次
decode       每次8 token
process runs 1
```

总计120/120进程成功。因为每个shape只有一个独立进程，本实验是宽覆盖survey，不把小差异写成
稳定排名。原始行、重算后的新schema summary、环境和比较表都在
[`076-data`](076-data/)；它们明确保存模型revision、shape、dtype、计时边界和失败。

精度政策也没有伪装成相同：microLLM是“BF16部分权重、FP32激活/其余路径”，PyTorch是完整
BF16模型。这里可以比较当前可执行系统，但不能把token相同当作全部logits相同。

## Prefill：短输入快，长batch明显掉队

下面是microLLM/PyTorch吞吐比；大于1表示microLLM更快：

| 模型 | T32 B1/B8 | T512 B1/B8 | T2048 B1/B8 |
|---|---:|---:|---:|
| Qwen | 1.392× / 2.097× | 1.703× / 0.615× | 0.540× / 0.173× |
| DeepSeek | 2.146× / 1.473× | 1.089× / 0.631× | 0.625× / 0.465× |

最清楚的失败是Qwen T2048 B8：microLLM约49.8k tok/s，PyTorch约287.7k tok/s，只剩
0.173×。相对自身B1，microLLM的B8扩展效率只有7.5%；DeepSeek对应为13.6%。短context的
绿色结果不能代表长prefill。

## Cached decode：大部分shape扩展好，DeepSeek长context仍落后

FP32 Cache下，B8相对PyTorch：

| 模型 | T32 | T512 | T2048 | microLLM B8效率（T2048） |
|---|---:|---:|---:|---:|
| Qwen | 3.218× | 2.113× | 1.180× | 95.2% |
| DeepSeek | 2.195× | 1.194× | 0.652× | 95.5% |

这说明batch scheduler的计算积木本身可以接近线性扩展，但DeepSeek T2048的单请求decode已经
慢于PyTorch，扩batch不会修复单slot Kernel/访存瓶颈。

## BF16 Cache：字节精确减半，不等于峰值减半

每个代表shape的KV Storage都精确缩小2倍。T2048 B8：

| 模型 | FP32 KV | BF16 KV | 引擎峰值 FP32→BF16 | decode BF16/FP32 |
|---|---:|---:|---:|---:|
| Qwen | 385.5 MiB | 192.75 MiB | 5.52→5.33 GiB | 1.040× |
| DeepSeek | 899.5 MiB | 449.75 MiB | 8.87→8.43 GiB | 1.161× |

Cache减半后，模型权重和临时activation仍然存在，所以总峰值只下降约3.4%和5.0%。BF16在
这些代表点没有变慢，DeepSeek长decode有所改善，但T2048 B8仍只有PyTorch的0.797×。

![Expanded inference service matrix](../assets/expanded-inference-service-matrix.svg)

## 正确性：成功运行不等于全部对齐

Qwen的18/18个FP32/BF16 cached比较都得到相同8-token suffix。DeepSeek只有10/18完全一致：

- T32 B1在第7个生成token分叉，B2/B4/B8却一致；
- T2048的FP32 B1/B2/B4/B8与BF16 B1/B8都在第4个token分叉；
- T512全部一致。

这不是被删掉的“噪声”。summary保存`matching_prefix_tokens`和
`first_token_difference`。当前两边驻留dtype政策不同，因此它首先证明“当前系统输出尚未
全shape对齐”，不能单凭本实验判断是哪一边错误。后续要用同dtype完整logits对照定位。

microLLM内部的FP32/BF16 Cache对照是12/12 suffix一致，所以这些分叉不是本轮BF16 Cache
新引入的；它们属于跨框架前向/精度政策差异。

## 这些数字没有测什么

- `kv_cache_utilization`是active/allocated字节，不是L2或硬件Cache命中率；
- `engine_current_bytes`在请求结束、KV对象销毁后读取，主要表示常驻模型，不是峰值瞬时占用；
- PyTorch行目前只有allocator peak，没有current、reserved和碎片；
- prefill/decode计时不包含加载、tokenizer、排队、网络和文本后处理，不能直接称完整TTFT；
- 当前只保存平均阶段时间，没有逐token P50/P95/P99；
- static batch使用同长度输入，不代表动态到达、不同长度、KV eviction、paging或多stream并发；
- 没有采集CU occupancy、HBM带宽、wave stall、功耗与joules/token；
- BF16只实测B1/B8，不能把FP32的B2/B4当作BF16结果；
- suffix相同不能替代完整logits max error、RMSE和top-k overlap。

## 快速CI与正式矩阵怎样配合

新增的C++测试每次CI都执行18个CPU和24个HIP tiny组合：三种context、B1–B8、FP32/BF16
Cache。HIP每行必须与CPU相同，Cache Storage字节必须等于手算公式。它在约0.2秒完成，用于
拦住功能回归；本页真实checkpoint矩阵用于发现性能和精度边界。两者不能互相替代。

## 决定

保留三套named suite、新显存/batch效率字段、首分叉字段和快速shape测试。下一轮优化优先级：

1. profile Qwen/DeepSeek T2048 B8 prefill，定位随batch恶化的Attention/临时量；
2. profile DeepSeek T2048 cached decode，先修B1热点再谈更多slot；
3. 用相同驻留dtype导出完整logits，定位DeepSeek第4个token分叉；
4. serving层继续做可变position slot refill，但不能拿它掩盖上述单batch Kernel问题。
