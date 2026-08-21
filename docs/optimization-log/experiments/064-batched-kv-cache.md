# Experiment 064 — batch-aware KV cache and cached decode

## 旧失败

Experiment 060中，Qwen/DeepSeek cached B2/B4/B8共6条在启动前明确`unsupported`。不能用
uncached结果冒充，也没有before Kernel timeline。

## 设计

- `KVCache(layers, capacity, batch)`记录共同batch和position；
- 每层Storage变成`[B,KVH,capacity,D]`，active view为`[B,KVH,T,D]`；
- full prefill按`batch×head`写capacity stride；
- step cache store、fused/long cached Attention都加入batch stride；
- model接受`[B,T]` prefill与`[B,1]` step；
- row-wise device argmax选择每行token。

B1构造器默认值和旧API保持兼容。

## 正确性

CPU测试使用两个不同prefix和不同next token，分别与独立full sequence的last logits对齐。
HIP测试覆盖B2 full prefill、继续step、KV shape/Storage和阶段内零host payload transfer。
官方矩阵48条token及KV理论/实际字节全部一致。

## 正式结果

context32，三进程中位数：

| 模型 | batch | micro tok/s | PyTorch | ratio | peak ratio |
|---|---:|---:|---:|---:|---:|
| Qwen | 1 / 2 / 4 / 8 | 91.9 / 182.9 / 363.2 / 721.1 | 122.2 / 242.3 / 490.7 / 984.8 | 0.75 / 0.75 / 0.74 / 0.73 | 1.15 / 1.13 / 1.09 / 1.03 |
| DeepSeek | 1 / 2 / 4 / 8 | 62.2 / 124.0 / 247.1 / 494.6 | 101.0 / 209.6 / 411.8 / 824.6 | 0.62 / 0.59 / 0.60 / 0.60 | 1.23 / 1.22 / 1.21 / 1.19 |

![Batched KV cache](../assets/batched-kv-cache.svg)

B8相对B1效率Qwen98.1%、DeepSeek99.5%。micro Cache为FP32，字节是PyTorch BF16动态
Cache的2.057×；batch越大，固定权重占比下降，peak ratio反而接近1。

## Retained profile

Qwen B8：715.6 tok/s、prepare32.3ms、end-to-end77.0ms。measured区只D2H 8次、256B；
D2D 1536次/12.6MB来自每层K/V prefill写入。top Kernel为cached Attention45.0ms、
full prefill Attention42.0ms、row argmax21.9ms。旧B8在启动前失败，不能伪造before trace。

## 决定

保留。cached B1/2/4/8能力和近线性扩展已建立。下一问题是Cache仍为FP32以及大量per-layer
D2D；下一节点应研究BF16 KV Cache，单独守住token/logit与长context精度门。
