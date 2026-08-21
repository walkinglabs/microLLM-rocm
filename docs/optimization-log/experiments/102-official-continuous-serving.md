# Experiment 102 — 让真实模型进入连续服务矩阵

前面的 scheduler 性能使用 tiny model，能解释机制，却不能回答 Qwen/DeepSeek 长上下文的
答案、显存和吞吐。这个节点不改 Kernel；它建立真实模型证据，并修复测试本身暴露的 Cache
容量问题。

## 假设与合同

假设：共享 Cache 若按当前请求所需的最大长度分配，而不是按模型理论最大长度分配，就能避免
短请求的巨大预留浪费，同时保持所有请求 token 不变。

合同：

- short/long context、2/4 slots、8/16 token 输出与 slot refill 都要出现；
- Qwen2.5-0.5B 与 DeepSeek-R1-Distill-Qwen-1.5B 使用固定 revision；
- 每个 case 三个 fresh process，warmup 不计时；
- 三次完整 `generated_tokens` 必须相同；
- Cache 分配必须精确等于模型 shape 推导公式；
- 保存 peak、resident、KV allocated/active、slot、transfer 和 scheduler counters；
- PyTorch 逐请求参考使用同一输入并逐 token 比较；
- 精度失败仍保存性能，不把性能通过写成总通过。

![Official continuous serving matrix](../assets/official-continuous-serving.svg)

## microLLM 结果

24/24 个独立进程通过，无 OOM、超时或进程内不确定。吞吐为三进程 min/p50/max：

| 模型 | case | tok/s min / p50 / max | KV MiB allocated / active | KV 利用率 | slot 利用率 |
|---|---|---:|---:|---:|---:|
| Qwen | short_s2 | 386.11 / 386.98 / 391.55 | 1.125 / 0.727 | 64.58% | 75% |
| Qwen | short_s4 | 572.46 / 763.21 / 777.34 | 3.750 / 2.215 | 59.06% | 75% |
| Qwen | long_s2 | 171.50 / 171.69 / 172.48 | 48.375 / 30.258 | 62.55% | 75% |
| Qwen | long_s4 | 414.36 / 414.57 / 415.82 | 96.750 / 72.703 | 75.15% | 100% |
| DeepSeek | short_s2 | 260.09 / 260.42 / 261.24 | 2.625 / 1.695 | 64.58% | 75% |
| DeepSeek | short_s4 | 463.83 / 476.25 / 479.95 | 8.750 / 5.168 | 59.06% | 75% |
| DeepSeek | long_s2 | 100.74 / 101.08 / 101.54 | 112.875 / 70.602 | 62.55% | 75% |
| DeepSeek | long_s4 | 239.93 / 240.04 / 240.17 | 225.750 / 169.641 | 75.15% | 100% |

`short_s4` Qwen 有一次明显低值，原始记录保留。长上下文 Cache 增长和峰值显存没有被隐藏：
Qwen long_s4 engine peak 为1917.54 MiB，DeepSeek为5213.01 MiB。

## PyTorch 精度门与比较边界

| 模型 | case | micro p50 | PyTorch sequential | 观察服务比 | token |
|---|---|---:|---:|---:|---|
| Qwen | short_s2 | 386.98 | 81.16 | 4.77× | exact |
| Qwen | short_s4 | 763.21 | 62.14 | 12.28× | exact |
| Qwen | long_s2 | 171.69 | 81.38 | 2.11× | exact |
| Qwen | long_s4 | 414.57 | 85.12 | 4.87× | exact |
| DeepSeek | short_s2 | 260.42 | 65.24 | 3.99× | exact |
| DeepSeek | short_s4 | 476.25 | 58.37 | 8.16× | **mismatch** |
| DeepSeek | long_s2 | 101.08 | 51.42 | 1.97× | **mismatch** |
| DeepSeek | long_s4 | 240.04 | 71.23 | 3.37× | **mismatch** |

PyTorch是逐请求串行、full-BF16；microLLM是连续slot，BF16 FFN/Attention/KV加部分FP32权重。
所以这些倍率不是同算法Kernel speedup。更重要的是，DeepSeek 3/4 未对齐，当前结论必须写成
`complete_with_recorded_accuracy_failures`。

## 结论

request-bound Cache假设得到支持：每个 case 的实分配与公式逐字节相等，短请求不再按32768容量
预留。官方 Qwen 连续服务矩阵通过；DeepSeek 长/高并发暴露稳定的跨框架 token 失败。

下一实验固定同一请求集，只扫1/2/4/8 slots，分离工作量变化与 batch 效率；精度侧从 DeepSeek
`short_s4` 的首个分叉 token 开始定位。

原始数据、合并表和环境见 [`102-data`](102-data/)。
