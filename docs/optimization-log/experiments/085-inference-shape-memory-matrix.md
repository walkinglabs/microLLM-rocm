# Experiment 085 — 每个decode token真的算了一次模型吗？

以前的推理矩阵已经有context和batch，但这次增加输出长度1/8/32时，一个很短的case把计时错误
照了出来：`new_tokens=1`只从prefill logits做argmax，没有执行任何cached Transformer forward。
把它叫“decode 1 token”会让吞吐看起来极高，却没有测到想测的东西。

## 先修尺子，再量速度

新的`steady`模式这样工作：

```text
计时外：prompt prefill → 选出seed token
计时内：seed进入模型 → 新logits → 选下一个token    # 1次forward
        新token进入模型 → 新logits → 再选token     # 又1次forward
```

因此JSON必须满足：

```text
measured_tokens == measured_forward_steps
KV active tokens == context + decode_tokens
```

旧pilot在T8 B1、N1得到microLLM 400.6 tok/s、PyTorch 21,588.5 tok/s；修正后冻结运行是
67.2/86.9 tok/s。旧数字测的主要是argmax，不是模型。旧pilot和运行中途重建造成的mixed-source
矩阵都保存在`085-data`并标记invalid，不能进入正式summary。

## 冻结实验合同

```text
GPU                 MI300X VF / gfx942
模型                Qwen2.5-0.5B、DeepSeek Distill Qwen 1.5B
context             8、512、2048
batch               1、8
steady decode steps 1、8、32
KV Cache            BF16；microLLM按同一sweep最大长度预留
warm-up             1次，不计时
measured            每进程3次
process runs        1（宽覆盖survey，不是稳定排名）
进程记录            72/72 pass
配对shape           36/36 greedy token完全相同
```

binary和runner在重跑前复制到`/tmp`冻结。Qwen使用GPU2，DeepSeek使用GPU1。第一份72行
矩阵后来确认`CMAKE_BUILD_TYPE`为空，所以只保留为语义、KV和显存survey，绝不用于发布速度。
随后使用同一冻结源码重建Release/gfx942，只对代表性的N8重跑24个进程记录。microLLM是部分
BF16权重加FP32激活路径，PyTorch是整模型BF16；因此仍是“当前可运行系统”比较，不是完全同dtype
的算子科学实验。

## 结果一：Release下Qwen全过线，DeepSeek长context仍未过线

下面是冻结Release binary，固定输出8步。数字是`microLLM / PyTorch`吞吐比，大于1才表示
microLLM更快：

| 模型 | T8 B1 / B8 | T512 B1 / B8 | T2048 B1 / B8 |
|---|---:|---:|---:|
| Qwen | 3.029× / 3.366× | 2.598× / 2.511× | 1.499× / 1.012× |
| DeepSeek | 2.372× / 2.142× | 1.674× / 1.450× | 0.866× / 0.671× |

Release的重要结论不是“全面领先”：Qwen六个点都过线，DeepSeek T8/T512过线，但T2048 B1
只有0.866×，B8只有0.671×。这把下一热点缩到DeepSeek long-context cached path。72行语义
survey的输出长度1/8/32都满足一token一forward，但因为它不是Release，不用于速度排序。

## 结果二：慢，但长batch更省峰值显存

下面使用Release N8；peak包含本次Cache prefill和decode，权重加载发生在更早阶段：

| 模型 | shape | microLLM peak | PyTorch peak | microLLM BF16 KV |
|---|---|---:|---:|---:|
| Qwen | T8 B8 | 1.18 GiB | 1.04 GiB | 1.50 MiB |
| Qwen | T512 B8 | 1.45 GiB | 3.44 GiB | 48.75 MiB |
| Qwen | T2048 B8 | 3.58 GiB | 10.68 GiB | 192.75 MiB |
| DeepSeek | T8 B8 | 4.19 GiB | 3.43 GiB | 3.50 MiB |
| DeepSeek | T512 B8 | 4.60 GiB | 5.94 GiB | 113.75 MiB |
| DeepSeek | T2048 B8 | 6.93 GiB | 13.59 GiB | 449.75 MiB |

短context时microLLM常驻策略更重；长context时原地softmax等已有生命周期优化让peak明显低于
PyTorch。这个结果只能说明引擎allocator看到的峰值，不含驱动和vendor-private内存。

## KV利用率为什么会变化

同一个shape sweep让microLLM按最长的32步预留草稿本：

```text
T8:    N1 22.5% → N8 40.0% → N32 100%
T512:  N1 94.3% → N8 95.6% → N32 100%
T2048: N1 98.5% → N8 98.8% → N32 100%
```

PyTorch DynamicCache只报告活跃前缀，所以每行是100%。Release N8矩阵只测一个输出长度，
两边都恰好预留到`context + 8`，利用率也是100%。这不是谁的硬件Cache命中率更高，而是
两种预留政策不同。summary同时保留allocated、active、capacity、每请求字节和KV占增量peak比例。

## 正确性没有被性能表吞掉

72行语义survey的36个shape全部greedy token一致。Release性能矩阵中，Qwen 6/6、DeepSeek
T8/T512 4/4一致；DeepSeek T2048 B1/B8都在第3个输出token分叉。这是已有的跨框架数值边界，
不是steady计时新引入。于是Release矩阵的结论是“12/12运行成功，10/12 token完全一致”，不能
把成功运行写成全部精度过线。

## 新测试门

- CPU：207/207；HIP：88/88；ASan/UBSan：200/200；
- tiny CPU矩阵：66个context/batch/dtype组合；tiny HIP矩阵：88个组合；
- context覆盖1、7、16、31/32/33、63/64/65、127/128；
- Cache检查allocated Storage、active view、position增长和Backing Storage地址不变；
- Python契约检查boundary suite、输出长度轴、KV公式、batch/context效率和进程P95。

## 下一步只优化一个热点

第一优先级是profile DeepSeek T2048 B8 Release steady decode，而不是继续优化短prompt。
当前microLLM每个decode
step还把选中的token ID搬回host做检查，N32、3次测量恰好产生96次D2H；PyTorch把selected
Tensor留在GPU。这是一个明确的同步候选。随后再区分cached Attention、QKV/O GEMM和allocator
各占多少时间；没有trace前不直接宣称需要哪一种FlashAttention实现。

![Steady decode throughput and memory](../assets/steady-inference-shape-memory.svg)

原始数据、冻结环境、invalid反例和机器可检查summary位于[`085-data`](085-data/)。
