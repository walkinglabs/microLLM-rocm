# Experiment 241 — P×V能不能直接写BF16 context

Status: `capability reject before timing`

## 归因

Experiment 240剩余的FP32→BF16每层1次，代码路径是FP32 Attention context进入BF16 O
projection。BF16→FP32的1次是grouped QKV的V进入当前FP32 Attention core。

本轮先测最小改动：保留FP32 probabilities、FP32 V、FP32 compute和原BTHD stride，
只把P×V的D output改为BF16。

## 能力门

| Descriptor | Retained FP32 | BF16 D status | Timing | Model route |
|---|---|---:|---|---|
| interleaved BTHD | pass | 6 | not started | none |
| zero-stride GQA BTHD | pass | 6 | not started | none |

![BF16 P×V output discard](../assets/bf16-pv-output-discard.svg)

两种descriptor都在计时前被当前hipBLASLt拒绝。这不是“慢”，而是当前输入/输出组合
没有可执行路径。

## 决定

- 撤回临时public API、layout dtype key和模型设想；
- 不枚举solution index，因为问题在descriptor能力层；
- 直接BF16 P×V output在当前backend关闭；
- 若继续删除O projection前cast，需要不同consumer/kernel，不是再调当前hipBLASLt路径。

证据：[`capability package`](../../../benchmarks/results/2026-08-25-bf16-pv-output-capability/)
