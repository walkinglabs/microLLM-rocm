# 推理矩阵：不要用一个短 prompt 代表所有推理

这份文档解释怎样公平地测量 microLLM 和 PyTorch 的推理。它故意使用简单词语；先看懂
“在测什么”，再看 tokens/s。

## 长上下文cached Attention自动策略

MI300X/gfx942上，已测Qwen H14/KV2/D64与DeepSeek H12/KV2/D128在BF16 KV、uniform cached
decode且prefix至少2048时，会自动使用保序的materialized-score路径。T512、FP32 KV、其他GPU、
其他head签名和positions-aware serving保持旧路径。

复现实验时可显式控制：

```bash
--cached-attention-materialized true \
--cached-attention-minimum-sequence 2048

# current对照
--cached-attention-materialized false
```

请同时保存JSON中的`cached_attention_materialized_policy`、`cached_attention_materialized_scores`和
`cached_attention_materialized_minimum_sequence`，不要只根据命令行猜测实际路径。

`split-P×V`是单独的显式研究路径，不属于Auto。它与materialized和完整split互斥：

```bash
--cached-attention-pv-splits 16 \
--cached-attention-minimum-sequence 2048 \
--cached-attention-materialized false
```

正式结果还要保存`cached_attention_pv_splits`与
`cached_attention_pv_minimum_sequence`。positions-aware serving继续使用专用参考路径。
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

脚本内置多套规模，避免每个人随手挑几个点：

| 套件 | context | batch | 输出长度 | 用途 |
|---|---|---|---|---|
| `smoke` | 8、128 | 1、2 | 1、4 | 几分钟内检查程序和JSON |
| `standard` | 8、32、128、512、2048 | 1、2、4、8 | 16 | 日常正式对比 |
| `serving` | 1、8、32、128、512、2048 | 1、2、4、8 | 1、8、32、64 | 服务短答、长答与batch效率 |
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
KV waste                    max(allocated - active, 0)，预留但还没写的页
KV share of incremental     KV allocated / incremental peak
non-KV incremental          max(incremental peak - KV allocated, 0)
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
两个worker各自`pass`也不等于聚合row通过：decode token分叉或prefill top token不同会把row
标成`precision_mismatch`，整体summary成为`complete_with_recorded_limits`。这些row的时间可以
帮助定位成本，但不能进入“精度对齐后的速度”结论。

如果microLLM和PyTorch都报告GPU不可见，runner把整体写成`invalid_environment`，只保留最先的
两个worker并以非零状态退出，不再为每个shape重复同一个环境错误。visible-device变量应只使用
一套逻辑；同一物理编号同时交给ROCR和HIP过滤可能造成二次过滤。

### 分叉后的轨迹不能直接比较

如果两种低精度在第3个token已经共同离开FP32，却到第9个token才彼此分开，那么三者第9步的
自然输入已经不同。此时比较logits是在比较三个问题。诊断CLI允许显式固定每次forward输入：

```bash
--decode-mode steady --use-cache true --workload decode \
--warmup 0 --steps 1 --new-tokens 3 \
--forced-decode-inputs 10,20,30
```

列表必须恰好等于`new-tokens`，每项必须在词表内。这个接口只用于一次zero-warmup cached诊断，
JSON会写`forced_decode_inputs`和数量；它不是生成API，也不能用于吞吐结论。

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
`kv_cache_waste_ratio=0.25`表示预留Cache里还有25%没有使用；这不是内存泄漏，而是固定容量策略
为后续token留下的空间。比较框架时必须同时看各自的`kv_cache_reservation_policy`。

仓库的MI300X 120条宽矩阵见
[Experiment 076](../optimization-log/experiments/076-expanded-inference-service-matrix.md)。其中旧prefill
行后来被确认是`full`语义，只作为完整logits证据保留；服务端修正版、profile和显存变化见
[Experiment 077](../optimization-log/experiments/077-serving-last-logit-prefill.md)。旧decode还把prefill
免费产生的首token计入吞吐；一token一forward、输出长度/KV利用率和冻结Release对照见
[Experiment 085](../optimization-log/experiments/085-inference-shape-memory-matrix.md)。
N64、B2/B4、显式KV waste字段、T2048/B2长上下文以及一次未稳定复现的batch-row失败见
[Experiment 095](../optimization-log/experiments/095-serving-inference-efficiency.md)。
Qwen3双计数fixture、64/64执行成功却有8个token分叉，以及修正后的诚实状态门见
[Experiment 364](../optimization-log/experiments/364-qwen3-fixture-shape-matrix.md)。
第一个T32/B1分叉的六policy完整logit与共同FP32 oracle见
[Experiment 365](../optimization-log/experiments/365-qwen3-bf16-first-divergence.md)。
五个唯一分叉状态、T512强制共同输入和T128反例见
[Experiment 366](../optimization-log/experiments/366-qwen3-bf16-oracle-sweep.md)。
T128反例的FFN/Attention/Cache三分归因见
[Experiment 367](../optimization-log/experiments/367-qwen3-bf16-t128-weight-islands.md)。
FFN分组、single、pair与repeat搜索见
[Experiment 368](../optimization-log/experiments/368-qwen3-bf16-ffn-layer-search.md)。
FFN gate/up/down七种scope和两个最小层集合见
[Experiment 369](../optimization-log/experiments/369-qwen3-bf16-ffn-projection-search.md)。
layers0–4 FP32候选的完整shape拒绝见
[Experiment 370](../optimization-log/experiments/370-qwen3-ffn0-4-fp32-reject.md)。
全模型gate-FP32五case预筛拒绝见
[Experiment 371](../optimization-log/experiments/371-qwen3-bf16-gate-fp32-reject.md)。
