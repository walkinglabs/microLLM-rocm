# Experiment 081 — 让probability覆盖已经无用的QK scores

长Attention当前顺序是：

```text
QK GEMM → scores [B,H,T,T]
scores  → causal softmax → probabilities [B,H,T,T]
probabilities → PV GEMM
```

softmax返回时，scores已经死亡，但旧实现会同时持有两份T² Tensor。Experiment 080说明现有
标量fused Kernel不能替代矩阵路径；本轮只删除这份确定的生命周期重叠。

## 单变量与alias安全

在HIP长序列library分支中，QK输出Tensor直接作为softmax input/output。register Kernel先完成
全行max读取，再完成每线程score读取与exp，本轮block reduction包含同步；只有分母就绪后才写
归一化结果。因此没有线程会覆盖另一个仍需读取的score，规约和数学顺序完全不变。

公共`causal_softmax()` reference仍然是out-of-place；只在内部Attention确认scores死亡后alias。

## 完整logits先过门

独立冻结的`ba0dec4` binary与候选在Qwen/DeepSeek T2048 B8分别比较151,936个logit：

```text
max abs = 0
RMSE    = 0
top     = equal
```

HIP T2048边界、mask严格为0、行和、MHA/GQA与CPU reference也通过。

## T2048 B8配对结果

| 模型 | reference peak | inplace peak | peak下降 | median pair throughput |
|---|---:|---:|---:|---:|
| Qwen | 5.147 GiB | 3.397 GiB | 34.0% | 1.017× |
| DeepSeek | 7.997 GiB | 6.496 GiB | 18.8% | 1.005× |

删除字节精确等于一份score Tensor：

```text
Qwen:     8 × 14 heads × 2048² × 4 bytes = 1,879,048,192
DeepSeek: 8 × 12 heads × 2048² × 4 bytes = 1,610,612,736
```

三对吞吐均未回退。Qwen一个reference pair较慢，但另外两对仍为1.017×/1.014×；DeepSeek
三对稳定1.004×–1.005×。本轮主指标是显存，吞吐只要求不倒退。

## Shape survey

两模型、T256/512/1024/2048、B1/B8共16点，top token全部一致，最差吞吐比0.990×。B8 peak：

| 模型 | T256 | T512 | T1024 | T2048 |
|---|---:|---:|---:|---:|
| Qwen peak下降 | 2.1% | 7.2% | 19.1% | 34.0% |
| DeepSeek peak下降 | 0.0% | 1.8% | 7.0% | 18.8% |

短shape由权重和其他activation主导，所以删除score Tensor不一定改变总peak；context/batch增大后
T²项逐渐成为峰值主因，结果与公式一致。

## Profile机制

Qwen相对Experiment 079：softmax `111.299→111.403ms`不变，全部Kernel
`481.707→476.962ms`，无profile forward `121.975→120.120ms`。logical allocation calls
`2172→2100`，正好少`24层 × 3 measured = 72`次。

DeepSeek allocation calls `2532→2448`，正好少`28 × 3 = 84`次。两模型Kernel calls、
copyBuffer和memory-copy没有增加；因此不是隐式复制，而是QK Tensor在同一stream被softmax
直接覆盖，再由PV GEMM读取。

最终门：CPU 196/196、HIP 85/85；ASan/UBSan 194/194与Torch-enabled 199/199继承自未改
host/Torch路径的父节点，优化日志和覆盖validator通过。

![In-place causal softmax](../assets/inplace-causal-softmax.svg)

## 决定

`keep`。它不新增模型能力、不改dtype、不改规约，只把已死亡Tensor的Storage复用为输出；
完整精度、paired吞吐、绝对字节公式和宽shape证据一致。

下一节点才考虑真正的online Attention。当前仍会物化一份`[B,H,T,T]` probabilities；彻底删除
它需要MFMA tile与online max/sum，不属于本轮生命周期优化。
