# Experiment 061 — batched long-prefill inference

## 假设

Experiment 060显示T512/1024 prefill只到PyTorch的0.026×–0.044×/0.009×–0.012×。
训练长Attention已经用batched hipBLASLt，因此最小假设是：普通inference复用同一公共
`causal_gqa_attention`，可以删除模型内部的可读QK/PV matmul。

## 第一次失败很重要

只在operator里加路由后，Qwen T512 `2011.23→2018.99 tok/s`，仅+0.39%。rocprof发现
模型根本没调用公共operator，而是在`Attention::forward_tensor`里重新手写repeat、QK、
softmax和PV。144次可读matmul占`629.41ms`、全部Kernel的78.39%。

于是第二版不再复制算法：模型直接调用公共causal GQA。T≥256由Experiment053/056的
strided-batched hipBLASLt执行，短序列走已有fused Kernel。

## 正式结果

| 模型 | context | before | after | 自身加速 | micro/PT | peak变化 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 512 | 2011 | 13522 tok/s | 6.72× | 0.308× | 不变 |
| Qwen | 1024 | 1063 | 14012 tok/s | 13.18× | 0.152× | +33.0% |
| DeepSeek | 512 | 1030 | 8654 tok/s | 8.40× | 0.229× | 不变 |
| DeepSeek | 1024 | 542 | 9073 tok/s | 16.73× | 0.156× | +12.2% |

![Batched long-prefill inference](../assets/batched-long-prefill-inference.svg)

四点top token一致，最大top-logit差0.195；T128也从2892.72提高到5159.83 tok/s，peak
不变。T1024必须诚实报告额外GQA head展开与临时T²表，不能只写速度。

## Profiler

```text
readable Attention matmul     144 calls / 629.41ms → 0
library GEMM                  +144 calls / +1.68ms
all Kernel                    802.89 → 156.87ms  5.12×
HIP API calls                 236,903 → 137,369  -42%
```

新的最大Kernel是causal softmax 39.44ms。整体仍只有PyTorch的15%–31%，因为混合精度
Linear/cast、softmax、布局和PyTorch融合路径仍不同。

## 决定

保留。这个节点同时删除了一份模型内Attention算法副本，使训练和推理共享数值合同。
下一步优先做full-sequence prefill-to-cache；它既解决4096 prompt分钟级准备，也避免服务
中先跑一次full prefill、再逐token重建相同Cache。
