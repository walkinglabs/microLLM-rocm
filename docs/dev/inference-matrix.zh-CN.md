# 推理矩阵：不要用一个短 prompt 代表所有推理

这份文档解释怎样公平地测量 microLLM 和 PyTorch 的推理。它故意使用简单词语；先看懂
“在测什么”，再看 tokens/s。

## 推理有三段，不是一件事

假设输入是一段128个token的文字，模型还要生成4个token：

```text
输入128个token ── prefill ── 得到第一个答案需要的信息
                                  │
                                  ├─ decode token 1
                                  ├─ decode token 2
                                  ├─ decode token 3
                                  └─ decode token 4
```

- **Prefill**：一次读完整个输入。矩阵通常较大，GPU容易吃饱。
- **Decode**：每次只生成一个新token。矩阵很窄，启动和Cache访问更重要。
- **End-to-end generate**：prefill和decode加在一起。它适合回答“用户等多久”，却不能说明
  慢的是哪一段。

本仓库把三者分开。cached decode 的计时器在 prompt 已经写进 Cache 后才开始；否则长
prompt会被错误算成“每个新token很慢”。

还有一个容易漏掉的细节：prefill本身已经给出了第一个token的logits。如果把这个token也算进
decode，那么“decode 1 token”只执行argmax，根本没有执行Transformer。正式runner使用
`--decode-mode steady`：先在计时外取得种子token，然后每个被计数的decode token都严格执行
一次模型forward。JSON中的`decode_step_semantics`是强制验收字段。

## Prefill 还要分“全部 logits”和“最后 logits”

语言模型训练需要每个位置的词表分数，因此完整前向输出`[B,T,V]`。生成服务只需要最后位置
决定第一个新token，输出`[B,1,V]`即可。如果把两者都叫prefill，会多做`T`倍左右的output
head工作，还可能把巨大的完整logits搬回CPU。

本仓库明确分成：

```text
--prefill-logits-mode last   # 默认，服务TTFT语义
--prefill-logits-mode full   # 显式完整logits/reference语义
```

microLLM分别调用`forward_inference_last_logits()`与`forward_inference()`；PyTorch的last路径
使用`logits_to_keep=1`。runner会把模式写进每条raw和summary，模式不同不能放在同一性能表。

## KV Cache 像一本预留页数的笔记本

生成新token时，模型不应每次重读所有旧token。KV Cache保存每层先前的K和V。

理论字节数是：

```text
2 × layers × batch × kv_heads × capacity_tokens × head_dim × dtype_bytes
```

开头的2表示K和V两份。必须同时报告：

- `allocated bytes`：底层Storage真正预留了多少；
- `active bytes`：当前已有token真正使用多少；
- `capacity tokens`：最多预留多少token；
- `active tokens`：已经写入多少token；
- `utilization = active / allocated`；
- `share of peak`：Cache占本次峰值显存多少；
- `peak share of device`：引擎峰值占整张卡总显存多少，它不是GPU算力利用率；
- `bytes per request`：batch增大后，每条请求平均承担多少Cache和峰值显存。

Tensor view只展示活跃前缀，不能拿它的`numel`冒充完整Storage。测试会检查：shape随token
增长，但Storage地址和预分配字节保持不变。

## 为什么要测多个context和batch

脚本内置三套规模，避免每个人随手挑几个点：

| 套件 | context | batch | 输出长度 | 用途 |
|---|---|---|---|---|
| `smoke` | 8、128 | 1、2 | 1、4 | 几分钟内检查程序和JSON |
| `standard` | 8、32、128、512、2048 | 1、2、4、8 | 16 | 日常正式对比 |
| `extended` | 1、8、32、128、512、1024、2048、4096 | 1、2、4、8、16 | 1、8、32 | 找极端边界、OOM和退化点 |
| `boundary` | 1、2、31/32/33、127/128/129、511/512/513、2048、4096 | 1、3、8 | 1 | 专门寻找tile边界与奇数batch错误 |

`--contexts`和`--batches`仍可覆盖套件，但正式报告会把最终使用的轴写进`summary.json`。
这些点主要回答两类问题：

| 轴 | 建议值 | 回答的问题 |
|---|---|---|
| context | 8、128、512、1024、2048、4096 | 输入变长后，Attention和Cache怎样增长？ |
| batch | 1、2、4、8 | 一次服务更多请求，吞吐和显存怎样变化？ |

输出长度不能被忽略。只生成1个token接近“首token后立刻停止”；8个token是短回答；32个token
可以看见固定启动开销被摊薄后的steady decode。可用`--decode-lengths 1,8,32`覆盖命名套件，
旧的`--decode-tokens 8`仍表示只测一个长度。

每个点再拆成prefill、cached decode、uncached decode。prefill不会因三个输出长度重复运行；
只有decode展开输出长度轴。框架不支持的组合必须写
`unsupported`；显存不够写`oom`。二者都不能被删除，也不能补一条模拟成功数据。

## 显存效率要把三本账分开

假设模型权重像书架，KV Cache像每位同学的草稿本，临时activation像桌面。只说“用了5 GB”
无法知道空间花在哪里，所以summary同时计算：

```text
peak                       本次运行到过的最高显存
resident weights            常驻权重
incremental peak            max(peak - resident weights, 0)
KV allocated                草稿本预留了多少页
KV active                   已经写了多少页
KV utilization              active / allocated
KV share of incremental     KV allocated / incremental peak
bytes per request           除以batch，比较每个请求的成本
tokens/s per peak GiB        每GiB峰值显存换来多少吞吐
```

microLLM默认让同一次输出长度sweep共享`context + 最大decode length`容量，PyTorch当前
DynamicCache随活跃前缀增长。因此两边
都要写`kv_cache_reservation_policy`，不能把“预留容量不同”误说成Kernel显存不同。每条成功
记录还必须满足：测量N个steady decode step后，真正写入Cache的是`context + N`。如果得到
`context + N - 1`，说明runner又把prefill免费得到的token混进了decode。这是一个很容易差一位
的边界测试。`--micro-cache-capacity exact`可以改为每个点只预留恰好需要的容量。

## 两边精度政策必须写明

当前可执行比较不是完全相同的驻留策略：

```text
microLLM：FFN/Attention权重为BF16，其余权重/激活仍含FP32路径
PyTorch：整模型BF16
```

因此结果会分别记录`precision_policy`与`resident_weight_bytes`。这个矩阵能比较“当前可用
路径”，却不能冒充完全同dtype的算子科学实验。

## 正确性门

每个成功decode点至少检查：

1. 同一进程多次生成token一致；
2. 相同batch行生成token一致；
3. cached与uncached token一致；
4. microLLM与PyTorch token是否一致；
5. KV理论/实际字节一致；
6. active Cache不能大于allocated Cache；
7. latency、吞吐、peak和整卡容量为有限正数；
8. batch吞吐扩展和效率以同context、同输出长度的B1作为基线；
9. 长context相对最短context的吞吐、延迟和peak变化分别报告；
10. 不同输出长度分别报告总延迟与每输出token延迟；
11. 参数量和revision没有改变。

跨框架token不一致不是“性能失败”，而是数值对齐失败。summary会保存相同前缀长度与首个
分叉位置；`-1`表示整段完全相同。不能只打印`false`，否则不知道第一步就错还是最后一步才分叉。

正式大模型矩阵之外，CI还有一套很小但真的执行计算的回归：

```text
tests/inference/shape_matrix_test.cpp
  CPU：11种context × 3种batch × FP32/BF16 Cache

tests/inference/hip_shape_matrix_test.cpp
  HIP：11种context × 4种batch × FP32/BF16 Cache，逐行对齐CPU
```

它还按模型层数、KV head、容量、batch和dtype重新计算Cache Storage字节，检查active view字节，
写入下一个token后再次检查position，并证明Storage地址没有悄悄改变。这样以后修改Kernel，某个
shape、预分配或低精度Cache做错时，不必等完整Qwen/DeepSeek跑分结束才发现。

## 运行方式

```bash
ROCR_VISIBLE_DEVICES=1 python3 \
  benchmarks/single_gpu/hf_inference_shape_matrix.py \
  --manifest /path/to/hf-models.local.json \
  --micro-binary build/apps/microllm_hf_infer \
  --pytorch-python /usr/local/bin/python3 \
  --output-directory /tmp/microllm-inference-matrix \
  --models qwen2.5-0.5b,deepseek-r1-distill-qwen-1.5b \
  --suite extended \
  --decode-lengths 1,8,32 --warmup 2 --steps 5 --runs 10
```

`raw.jsonl`保存每个新进程；`summary.json`只对完整成功的配对取中位数。正式结论还要记录
GPU、ROCm、PyTorch/Transformers版本和失败行。正式看P95时建议至少10个独立进程；这里的
`process_latency_ms_p95`是各进程平均请求时间的P95，不是假装成逐token服务P99。
`batch_efficiency=1`表示吞吐随batch理想线性
增长；例如B4吞吐是B1的3.2倍，则效率是`3.2 / 4 = 0.8`，也就是80%。

仓库的MI300X 120条宽矩阵见
[Experiment 076](../optimization-log/experiments/076-expanded-inference-service-matrix.md)。其中旧prefill
行后来被确认是`full`语义，只作为完整logits证据保留；服务端修正版、profile和显存变化见
[Experiment 077](../optimization-log/experiments/077-serving-last-logit-prefill.md)。旧decode还把prefill
免费产生的首token计入吞吐；一token一forward、输出长度/KV利用率和冻结Release对照见
[Experiment 085](../optimization-log/experiments/085-inference-shape-memory-matrix.md)。
