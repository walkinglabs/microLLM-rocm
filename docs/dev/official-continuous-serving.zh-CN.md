# 官方 Qwen / DeepSeek 连续推理测试：怎样读懂速度与显存

这一页不要求你懂 GPU。先把推理服务想成一家只有几张桌子的餐馆：

- 一条请求是一位客人；
- prompt 是客人已经写好的问题；
- 每生成一个 token，就像服务员再写一个字；
- slot 是一张正在使用的桌子；
- KV Cache 是放在桌上的草稿纸，模型不用每次把前文重算一遍。

`ContinuousBatchScheduler`允许先完成的客人离开，再让等待的客人坐到同一张桌子。测试不能只问
“最快一次是多少”，还要问答案对不对、桌子是否真的被利用、草稿纸是否浪费，以及长问题是否
让显存失控。

## 1. 测了哪些情况

`hf_continuous_matrix.py`当前固定四组工作量，并对 Qwen2.5-0.5B 与
DeepSeek-R1-Distill-Qwen-1.5B 各运行三次独立进程：

| case | slot | 请求数 | prompt 长度 | 每条生成长度 | 想回答的问题 |
|---|---:|---:|---|---|---|
| short_s2 | 2 | 4 | 8、8、32、32 | 8/16 混合 | 两张桌子能否补位 |
| short_s4 | 4 | 8 | 8–64 | 8/16 混合 | 短请求增加并发后的吞吐 |
| long_s2 | 2 | 4 | 512、512、2048、2048 | 8/16 混合 | 长前文、低并发的代价 |
| long_s4 | 4 | 8 | 256–2048 | 8/16 混合 | 长前文、高并发与显存峰值 |

每个进程先热身一次，再测三轮。热身时间不进入吞吐。三次进程都保存完整 token 数组，而不只
保存一个看不见内容的总分。

## 2. 为什么新增 Cache 容量上限

模型配置允许 32768 token，不代表本次请求真的需要这么长。若两个 slot 一开始就各预留模型
最大长度，一次几十 token 的小测试也可能分配几百 MiB。

调度器现在接受 `max_sequence_length`。官方 runner 把它设成这批请求中最大的
`prompt_length + new_token_length`。BF16 KV Cache 的理论字节数是：

```text
层数 × 2(K和V) × slot数 × 容量 × KV头数 × 每头宽度 × 2字节
```

测试会用模型配置重新计算这个数，并要求程序报告的 `allocated_cache_bytes` 精确相等。超过模型
上限的请求会在提交时直接报错，不会悄悄写越界。

## 3. 输出中的指标是什么意思

- `tokens_per_second`：排除热身后，每秒真正生成多少 token；
- `engine_peak_bytes`：microLLM 分配器观察到的峰值显存；
- `resident_weight_bytes`：常驻权重占用；
- `allocated_cache_bytes`：为所有 slot 预留的草稿纸；
- `peak_active_cache_bytes`：最忙时真正有内容的草稿纸；
- `kv_cache_byte_utilization`：有效草稿纸 / 预留草稿纸；
- `slot_utilization`：整个调度过程中桌子真正有人使用的比例；
- `slot_refills`：旧请求结束后，有多少新请求复用了空位；
- `generated_tokens`：每条请求实际生成的 token，可逐项与参考实现核对；
- H2D/D2H/D2D：主机到显卡、显卡到主机、显卡内部复制的次数与字节。

## 4. MI300X 实测结果

microLLM 三次独立进程的中位数如下：

| 模型 | case | tok/s p50 | KV 预留 / 峰值有效 | KV 利用率 | slot 利用率 | engine 峰值 |
|---|---|---:|---:|---:|---:|---:|
| Qwen | short_s2 | 386.98 | 1.125 / 0.727 MiB | 64.58% | 75% | 1206.37 MiB |
| Qwen | short_s4 | 763.21 | 3.750 / 2.215 MiB | 59.06% | 75% | 1215.56 MiB |
| Qwen | long_s2 | 171.69 | 48.375 / 30.258 MiB | 62.55% | 75% | 1559.81 MiB |
| Qwen | long_s4 | 414.57 | 96.750 / 72.703 MiB | 75.15% | 100% | 1917.54 MiB |
| DeepSeek | short_s2 | 260.42 | 2.625 / 1.695 MiB | 64.58% | 75% | 4287.63 MiB |
| DeepSeek | short_s4 | 476.25 | 8.750 / 5.168 MiB | 59.06% | 75% | 4305.43 MiB |
| DeepSeek | long_s2 | 101.08 | 112.875 / 70.602 MiB | 62.55% | 75% | 4746.53 MiB |
| DeepSeek | long_s4 | 240.04 | 225.750 / 169.641 MiB | 75.15% | 100% | 5213.01 MiB |

Qwen `short_s4` 三次为 572.46、763.21、777.34 tok/s，有一次明显慢。原始值被保留；报告使用
中位数，不把最快值包装成稳定速度。

## 5. 与 PyTorch 到底对齐了什么

PyTorch 程序接收完全相同的 prompt token 和生成长度，使用官方 Transformers 模型、BF16、
greedy decoding 与 KV Cache。它按请求逐条运行，因此是 token 正确性参考和框架级串行基线，
不是与 slot scheduler 相同的算法。

| 模型 | short_s2 | short_s4 | long_s2 | long_s4 |
|---|---|---|---|---|
| Qwen | 完全一致 | 完全一致 | 完全一致 | 完全一致 |
| DeepSeek | 完全一致 | **不一致** | **不一致** | **不一致** |

所以当前可以说：Qwen 4/4 工作量逐 token 对齐；DeepSeek 只有 1/4 对齐。不能说 DeepSeek 长
上下文已达到 PyTorch 精度。DeepSeek 三个失败会成为下一轮定位的硬门。

microLLM/PyTorch 的观察吞吐比为 1.97×–12.28×，但它比较的是“连续 slot 服务”和“逐请求
串行服务”，两边权重驻留策略也不同。这个数只能描述当前两个程序处理同一请求集的服务吞吐，
不能写成同算法 Kernel 加速比。

## 6. 怎样复现

```bash
python3 benchmarks/single_gpu/hf_continuous_matrix.py \
  --manifest /path/to/model-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory run/continuous \
  --suite standard --warmup 1 --steps 3 --runs 3
```

PyTorch 参考用 `pytorch_continuous_reference.py` 对每组 case 单独运行。之后用
`compare_hf_continuous.py`合并两边 JSON。完整原始证据见
[Experiment 102](../optimization-log/experiments/102-official-continuous-serving.md)。

## 7. 下一步还缺什么

这四组已经覆盖短/长、2/4 slot 和补位，但 2-slot 与 4-slot 的请求集并不完全相同。因此下一组
实验要固定同一批请求，只改变 slot 为 1/2/4/8，才能得到公平的 batch 效率曲线。之后再定位
DeepSeek 首个分叉 token，并补 P50/P95 请求延迟；在这些门通过前，不宣称生产级服务。
