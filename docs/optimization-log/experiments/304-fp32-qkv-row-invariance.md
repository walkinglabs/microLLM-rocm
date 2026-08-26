# Experiment 304：Q只有一个跨M位级solution，K/V有五个

Status: complete-model QKV gate selected

## 使用真实Forward Descriptor

固定2048-row FP32输入块，重复成M2048/4096/8192/16384。K=1536，分别测试Q的N1536和K/V的
N256。候选必须四个M共同support、CPU sentinel通过，并让所有2048-row输出块位级相同。

![FP32 QKV row invariance](../../../benchmarks/results/2026-08-26-fp32-qkv-row-invariance/qkv-row-invariance.svg)

| Operation | 共同候选 | 位级候选 | 选择 | workspace | 四个M Event和 |
|---|---:|---:|---:|---:|---:|
| Q | 12 | 1 | 296100 | 0 | 2.23810 ms |
| K/V | 22 | 5 | 292135 | 0 | 1.11366 ms |

K/V另外四个exact候选是291992、292147、297273和300609。所有共同候选都通过CPU sentinel。
非保序候选的最大block差异看起来很小：Q 2.14e-7、K/V 3.58e-7；Experiment 303已经证明这种
差异经过RoPE/BF16 cache可以变成0.03125级边界变化。

第一次实现编译在measurement前失败：本机hipBLASLt `getIndexFromAlgo`要求mutable引用，inventory
错误使用const。修正并通过CTest后才运行正式数据。

## 决定

不设默认。下一步只在完整DeepSeek full-prefill显式注册Q=296100、K/V=292135，比较B1/2/4/8：

- Block0原始BF16 K/V prefix；
- step0完整151,936 logits与相同行；
- prefill/decode吞吐和peak；
- registry hit/miss/dispatch。

若cache与logits显著收敛且性能可接受，再做Qwen/多context边界；否则拒绝version-local方案。

证据：[`FP32 QKV row invariance`](../../../benchmarks/results/2026-08-26-fp32-qkv-row-invariance/)
