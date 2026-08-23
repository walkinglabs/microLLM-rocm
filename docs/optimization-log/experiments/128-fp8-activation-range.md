# Experiment 128：同一把activation尺子既太短，又太粗

## 问题

weight已经每个Tensor一把尺子，但所有Linear输入仍共用activation scale 0.2。E4M3-FNUZ的有限
幅值是240，因此这把尺子能表示约`±48`。

如果activation绝对值大于48，它会撞到尺子末端；如果绝对值通常只有2–8，0.2的格子又太粗，
大量FP8格子没有利用。

## 先修一个观测失败

第一次Qwen worker生成317条trace，但runner只匹配72/96。缺的是每层FP32 FFN的
`ffn.activated`：BF16 diagnostics会记录，普通FP32路径没有。合同立即停止，DeepSeek未运行，
失败trace原样保留。

只在`record_all_layer_details` opt-in时补该观测点；默认block-0 trace计数不变。单测从失败变为
通过后，Qwen 96行pilot先过合同，再运行正式两模型。

## 正式结果

两个worker得到Qwen96行、DeepSeek112行，共208个层级输入边界。16行可能饱和，全部来自FFN。

| 模型/边界 | 层数 | amax min/P50/max | 超过±48的层 | 最大层 | max/range |
|---|---:|---:|---:|---:|---:|
| Qwen attention norm | 24 | 0.74 / 31.50 / 46.59 | 0 | 6 | 0.97× |
| Qwen attention context | 24 | 0.13 / 2.59 / 8.59 | 0 | 21 | 0.18× |
| Qwen FFN norm | 24 | 6.59 / 33.92 / 336.26 | 4 | 3 | 7.01× |
| Qwen FFN activated | 24 | 2.50 / 5.56 / 1723.77 | 4 | 21 | 35.91× |
| Deep attention norm | 28 | 4.74 / 7.10 / 11.31 | 0 | 27 | 0.24× |
| Deep attention context | 28 | 1.12 / 2.97 / 9.84 | 0 | 27 | 0.20× |
| Deep FFN norm | 28 | 2.50 / 11.34 / 75.00 | 3 | 1 | 1.56× |
| Deep FFN activated | 28 | 3.14 / 10.55 / 3081.23 | 5 | 2 | 64.19× |

![FP8 activation range](../assets/fp8-activation-range.svg)

Qwen饱和层：FFN norm 3/5/20/21，activated 2/3/5/21。DeepSeek：norm 1/2/5，activated
1/2/25/26/27。

## 推翻了什么

“找到一个更大的全局activation scale就够了”被推翻。把scale放大能容纳少数3000级异常值，
却让P50只有2.6–10.5的大多数context/activated输入只使用十几到几十个正数格子；把scale缩小
又会截断16个FFN边界。

## 下一步设计

证据先支持`per Linear-input Tensor` device amax，而不是直接声称必须per-row/per-token：

```text
input Tensor on GPU
→ device amax
→ scale Tensor = max(amax / 240, minimum scale)
→ device quantize
→ hipBLASLt scaled GEMM
```

第一版允许每个Linear一次device reduction，但不得每层D2H或全局同步。先测完整logits和新增
Kernel/同步开销；只有同一个输入Tensor内部不同row/token范围仍冲突时，才升级per-row/per-token。
