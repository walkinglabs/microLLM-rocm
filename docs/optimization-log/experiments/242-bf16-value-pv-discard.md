# Experiment 242 — 保留BF16 V能不能删掉反向cast

Status: `capability reject before timing`

本轮只把V从FP32换成grouped QKV已有的BF16输出，probabilities、compute和context输出仍是
FP32。普通interleaved BTHD和zero-stride GQA都返回hipBLASLt status 6。

| Descriptor | BF16 V support | Timing | Model route |
|---|---:|---|---|
| interleaved BTHD | status 6 | not started | none |
| zero-stride GQA | status 6 | not started | none |

![BF16 V P×V discard](../assets/bf16-value-pv-discard.svg)

临时API和实现已撤回，原FP32 conformance重新通过。结合Experiment 241，剩余一进一出两个
cast都不能靠当前hipBLASLt mixed-dtype descriptor直接消除。若重开，需要不同kernel架构。

证据：[`capability package`](../../../benchmarks/results/2026-08-25-bf16-value-pv-capability/)
