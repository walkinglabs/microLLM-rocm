# Experiment 062 — full-sequence prefill-to-KV-cache

## 问题

Experiment 060把steady decode和prompt准备分开后，发现Cache准备仍逐token重放模型：
Qwen/DeepSeek T1024一次warm-up约38.5/54.9秒，T2048约115.6/171.2秒。它不是decode慢，
而是没有full-sequence prefill-to-cache API。

## 设计

新增B1 API：

```text
forward_prefill_cached([1,T] tokens, empty cache)
  → 每层一次计算Q/K/V与full causal Attention
  → K/V按head复制到[1,KVH,capacity,D] Storage
  → active view变成[1,KVH,T,D]
  → cache.position一次advance(T)
  → 只返回last-token logits [1,1,V]
```

`hf_infer`默认`--cache-prefill-mode full`；显式`token`只保留为失败复现/reference。
公共`inference::generate()`也使用full prefill，不是只优化CLI；prompt H2D从每token一次变成
整段一次，总Int32字节保持不变。

## 两次被测试拦下的错误

第一版把紧排`[head,T,D]`整块复制到按`capacity`跨步的Storage。full prefill logits正确，
但继续第4个token的16个logits最大偏差0.153；测试指出head1写错偏移。最终按head分别D2D。

第二版返回整个`[T,V]` logits，速度快却让T1024峰值增加33%/12%。生成只需要最后一行，
合同改成`[1,1,V]`后，最终峰值代价降到Qwen+11.5%、DeepSeek+3.4%。

## 正式结果

三进程中位数，时间单位ms：

| 模型 | T | prepare micro/PT | end-to-end micro/PT | peak vs旧token | token |
|---|---:|---:|---:|---:|---|
| Qwen | 8 | 13.0 / 11.3 | 44.9 / 45.1 | 1.000× | 一致 |
| Qwen | 512 | 37.2 / 11.8 | 130.8 / 43.8 | 1.034× | 一致 |
| Qwen | 1024 | 71.3 / 13.2 | 228.2 / 52.6 | 1.115× | 一致 |
| DeepSeek | 8 | 19.0 / 14.0 | 68.9 / 54.8 | 1.000× | 一致 |
| DeepSeek | 512 | 57.9 / 14.2 | 203.6 / 52.1 | 1.011× | 一致 |
| DeepSeek | 1024 | 108.8 / 17.5 | 351.2 / 57.7 | 1.034× | 一致 |

![Full-sequence prefill to KV cache](../assets/full-prefill-kv-cache.svg)

T2048一进程prepare为156.5/231.4ms，不再是分钟级；peak代价为旧token的1.405×/1.111×。

## Profiler

同一Qwen T512二进制，仅切换CLI模式：

```text
prepare                 10360.6 → 37.7ms    274.8×
end-to-end              10392.8 → 69.7ms    149.0×
Kernel calls            497177 → 3201        155× fewer
Kernel time             20.276 → 0.180s      112.3×
HIP API calls           4552067 → 129885      35× fewer
```

旧路径仅cached Attention就有24624次、13.07s。

## 决定

保留。B1长prompt从不可服务变成毫秒级；剩余端到端差距来自microLLM混合驻留、FP32 Cache、
steady cached Attention与PyTorch融合。下一节点是batch-aware KV Storage与cached B2/B4/B8。
