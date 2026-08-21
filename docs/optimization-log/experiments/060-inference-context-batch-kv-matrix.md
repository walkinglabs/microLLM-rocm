# Experiment 060 — inference context, batch and KV-cache matrix

## 为什么重新测

Experiment 036 的固定短prompt矩阵曾得到4/4高于PyTorch。它证明了当时那几个点，不等于
长context、batch或服务场景。用户要求增加短/长上下文、batch、KV Cache显存和多步计时；
这次实验的首要目标是尝试推翻旧结论，而不是再找一个绿色点。

## 先修benchmark，再看速度

subagent在物理GPU1写出第一版runner，并在GPU1实跑。审查发现四个问题：

1. active Tensor view字节被误叫作底层allocated Storage字节；
2. 由错误字节反推出不存在的3-byte Cache元素；
3. microLLM cached计时包含prompt ingestion，PyTorch边界不一致；
4. PyTorch使用高层`generate()`，两边greedy循环语义不同。

这些pilot完整保留在`060-data/invalid-pilots`并标为invalid。修正版使用：

```text
prefill timer   = 完整prompt forward
cache prepare   = prompt写入KV Cache，单独报告
cached decode   = cache准备后，只计新token
uncached decode = 每个新token重算完整prefix
```

microLLM的Cache allocated读取`Storage::num_bytes()`；active读取当前prefix view。模型测试
证明shape随token增长，但Storage地址和容量不变。

## 精度边界

当前两条可执行路径并非完全同dtype：

```text
microLLM = BF16 FFN/Attention weights + remaining FP32 paths
PyTorch  = full model BF16
```

所以每行同时保存`precision_policy`与`resident_weight_bytes`。这是当前系统对比，不冒充
完全同dtype的算子实验。Qwen/DeepSeek驻留权重比分别为`1.276×/1.263×`。

## 核心三进程矩阵

MI300X物理可见卡1；warm-up 1次但不计时，measured 2次，三个独立进程中位数：

| 模型 | context | prefill micro/PT | cached decode micro/PT | Cache相对自身uncached | token |
|---|---:|---:|---:|---:|---|
| Qwen 0.5B | 8 | 0.649× | 1.049× | 2.64× | 一致 |
| Qwen 0.5B | 128 | 0.253× | 0.762× | 4.58× | 一致 |
| Qwen 0.5B | 512 | 0.044× | 0.318× | 10.45× | 一致 |
| DeepSeek 1.5B | 8 | 0.510× | 0.845× | 2.45× | 一致 |
| DeepSeek 1.5B | 128 | 0.195× | 0.588× | 4.58× | 一致 |
| DeepSeek 1.5B | 512 | 0.026× | 0.267× | 14.43× | 一致 |

![Inference context, batch and KV matrix](../assets/inference-context-batch-matrix.svg)

旧“4/4 parity可推广”的解释被推翻。Cache本身很重要，但microLLM的prefill和长cached
Attention仍远慢于PyTorch。

## 有warm-up的长上下文

物理可见卡3；一轮warm-up排除在一轮measured之外：

| 模型 | context | prefill micro/PT | cached decode micro/PT | Cache相对自身uncached |
|---|---:|---:|---:|---:|
| Qwen | 1024 | 0.0116× | 0.2037× | 35.2× |
| Qwen | 2048 | 0.00435× | 0.1262× | 81.4× |
| DeepSeek | 1024 | 0.00947× | 0.1658× | 48.1× |
| DeepSeek | 2048 | 0.00381× | 0.0899× | 100.6× |

所有decode token仍一致。另一个零warm-up探索覆盖4096，但首次初始化让框架比值方向错误，
因此只用于证明可运行和Cache容量，不用于速度排名。4096逐token建立prompt Cache达到分钟级，
成为稳定架构失败。

## Batch矩阵

物理可见卡3、context32、单进程探索：

| 模型 | B1→B8 prefill效率 | B1→B8 uncached decode效率 | cached B2/B4/B8 |
|---|---:|---:|---|
| Qwen | 80.4% | 36.7% | unsupported |
| DeepSeek | 55.8% | 38.4% | unsupported |

42条成功，6条microLLM cached batch明确`unsupported`；PyTorch对应行成功。unsupported不是
慢，也不能用uncached结果代替。

## KV Cache显存

microLLM Cache是FP32（4 bytes），PyTorch当前路径是BF16（2 bytes）。microLLM allocated：

| 模型 | T8 | T128 | T512 | T1024 | T2048 |
|---|---:|---:|---:|---:|---:|
| Qwen | 0.28 MiB | 3.09 MiB | 12.09 MiB | 24.05 MiB | 48.05 MiB |
| DeepSeek | 0.66 MiB | 7.22 MiB | 28.22 MiB | 56.11 MiB | 112.11 MiB |

利用率从T8的91.67%逐步接近100%，因为只预留`prompt + requested new tokens`，最后一个
生成token不需要再次forward。最终schema smoke还分开报告：Qwen T8 Cache准备
`82.51ms vs 12.63ms`，steady decode接近，端到端却是`93.26ms vs 23.74ms`。

## 决定

保留benchmark、CLI、测试和诚实失败；废除把Experiment 036当作总体推理parity证据的
展示方式。正确性证据仍有效，性能结论被新矩阵取代。

下一节点按证据顺序：

1. 将已验证的strided-batched hipBLASLt QK/PV接入长prefill inference；
2. 新增full-sequence prefill-to-KV-cache，取消逐token prompt准备；
3. KV Cache增加batch维与cached B2/B4/B8；
4. batch argmax留在GPU；
5. 建立完全同驻留dtype的科学对照。
