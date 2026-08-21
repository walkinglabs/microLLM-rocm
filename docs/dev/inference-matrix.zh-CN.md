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

| 套件 | context | batch | 用途 |
|---|---|---|---|
| `smoke` | 8、128 | 1、2 | 几分钟内检查程序和JSON |
| `standard` | 8、32、128、512、2048 | 1、2、4、8 | 日常正式对比 |
| `extended` | 1、8、32、128、512、1024、2048、4096 | 1、2、4、8、16 | 找极端边界、OOM和退化点 |

`--contexts`和`--batches`仍可覆盖套件，但正式报告会把最终使用的轴写进`summary.json`。
这些点主要回答两类问题：

| 轴 | 建议值 | 回答的问题 |
|---|---|---|
| context | 8、128、512、1024、2048、4096 | 输入变长后，Attention和Cache怎样增长？ |
| batch | 1、2、4、8 | 一次服务更多请求，吞吐和显存怎样变化？ |

每个点再拆成prefill、cached decode、uncached decode。框架不支持的组合必须写
`unsupported`；显存不够写`oom`。二者都不能被删除，也不能补一条模拟成功数据。

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
8. batch吞吐扩展和效率以同context的B1作为基线；
9. 参数量和revision没有改变。

跨框架token不一致不是“性能失败”，而是数值对齐失败。summary会保存相同前缀长度与首个
分叉位置；`-1`表示整段完全相同。不能只打印`false`，否则不知道第一步就错还是最后一步才分叉。

正式大模型矩阵之外，CI还有一套很小但真的执行计算的回归：

```text
tests/inference/shape_matrix_test.cpp
  CPU：3种context × 3种batch × FP32/BF16 Cache

tests/inference/hip_shape_matrix_test.cpp
  HIP：3种context × 4种batch × FP32/BF16 Cache，逐行对齐CPU
```

它还按模型层数、KV head、容量、batch和dtype重新计算Cache Storage字节。这样以后修改Kernel，
某个shape或低精度Cache做错时，不必等完整Qwen/DeepSeek跑分结束才发现。

## 运行方式

```bash
ROCR_VISIBLE_DEVICES=1 python3 \
  benchmarks/single_gpu/hf_inference_shape_matrix.py \
  --manifest /path/to/hf-models.local.json \
  --micro-binary build/apps/microllm_hf_infer \
  --pytorch-python /usr/local/bin/python3 \
  --output-directory /tmp/microllm-inference-matrix \
  --models qwen2.5-0.5b,deepseek-r1-distill-qwen-1.5b \
  --suite standard \
  --decode-tokens 4 --warmup 1 --steps 2 --runs 3
```

`raw.jsonl`保存每个新进程；`summary.json`只对完整成功的配对取中位数。正式结论还要记录
GPU、ROCm、PyTorch/Transformers版本和失败行。`batch_efficiency=1`表示吞吐随batch理想线性
增长；例如B4吞吐是B1的3.2倍，则效率是`3.2 / 4 = 0.8`，也就是80%。

仓库当前的MI300X 120条实测、图和失败分析见
[Experiment 076](../optimization-log/experiments/076-expanded-inference-service-matrix.md)。
