# Experiment 094 — 空座位终于能装入新 prompt

Experiment 093让共享Cache里的不同row按不同position继续decode，但新请求仍无法进入一个刚清空的
row。本节点只补这个缺口，不同时修改scheduler或并行Kernel。

## 假设与验收门

假设：复用已验证的B1 full-prefill，再把有效K/V前缀D2D复制进共享Storage的目标row，可以建立
一个简单、可独立对照的slot admission oracle。

必须同时满足：

1. prefill logits与独立B1一致；
2. 未替换row的所有layer K/V逐项不变；
3. 共享Cache backing地址不变；
4. 目标position从0变成prompt长度，其他position不变；
5. 下一次divergent decode中两行继续与各自独立B1一致；
6. FP32/BF16、CPU/HIP都通过，HIP执行区间D2H payload calls为0；
7. 非空row和越界row明确失败。

## 保留的实现

```text
shared positions [0,3]
       │
       ├─ prompt [1,2] → existing B1 full prefill → temporary B1 K/V
       │                                      │
       │                     same-device copy by layer/head
       │                                      ▼
       └──────────────────────────── shared row 0

result positions [2,3] → divergent decode → [3,4]
```

当新prompt比其他row更长时，公共logical prefix先扩到两者最大值；底层capacity不改变。全空的共享
Cache也可以先给任意一个row prefill，Storage由本接口建立。

![Shared-cache row prefill](../assets/slot-row-prefill.svg)

## 结果

| 门 | 结果 |
|---|---:|
| 完整CPU/HIP配置 | 302/302 |
| CPU-labelled | 211/211 |
| HIP-labelled | 91/91 |
| ASan/UBSan CPU | 204/204 |
| Cache dtype | FP32、BF16 |
| 状态转移 | `[3,3]→[0,3]→[2,3]→[3,4]` |
| 未替换row | 每层K/V逐项不变 |
| HIP执行D2H payload | 0 calls |

## 能得出什么，不能得出什么

可以得出：模型层现在拥有“清空某row→写入新prompt→两row从不同position继续”的完整正确性积木。

不能得出：continuous scheduler已经完成，或动态batch已经加速。当前路径临时分配B1 Cache，并对
每层每个KV head做D2D copy；它尚未测真实模型slot-refill吞吐。下一个节点应先让scheduler使用本
oracle跑通延迟到达、提前结束和重复补位，再单独优化copy和positions-aware Kernel。

机器可读证据见 [`094-data`](094-data/)。
