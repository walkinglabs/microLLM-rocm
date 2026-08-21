# 1、2、4、8 个 slot：怎样公平测试 batch 效率

## 1. 公平比较是什么意思

如果两张桌子接待4位客人，四张桌子却接待8位客人，那么总速度变快可能只是因为工作量不同。
公平的 slot 测试要把客人和问题全部固定，只改变同时服务的人数。

本测试固定两批各8条请求：

- short：prompt 为8、8、16、16、32、32、64、64 token；
- long：prompt 为256、256、512、512、1024、1024、2048、2048 token；
- 每条输出8或16 token；
- 同一批请求分别使用1、2、4、8个slot；
- 每种配置运行三个独立进程，报告min/p50/max。

## 2. 第一轮为什么失败

第一轮48个进程只有30个通过。两个模型的`short_s1`、`long_s1`和`long_s2`都稳定失败，
错误是“KV prefix需要空Cache”。

问题不是显存不够。请求结束后，Cache的逻辑位置已经回到0，但为了下次少申请显存，底层Storage
仍然保留。程序错误地把“抽屉还在”理解成“抽屉里还有旧作业”，于是拒绝新prompt。

修复只改变一条选择规则：

```text
Storage从未创建 → 首次prefill快路径
Storage存在但row位置为0 → 复用Storage并覆盖新prefix
```

CPU和HIP测试都加入单slot、不同prompt长度、连续两次refill。随后原样重跑48个进程，48/48
执行通过；旧的18条失败没有删除。

## 3. 速度怎样算

以S1中位吞吐为基线：

```text
speedup = Sx吞吐 / S1吞吐
parallel efficiency = speedup / x
```

例如S4速度是S1的2.4倍，效率就是2.4÷4=60%。100%表示每多一张桌子都得到同样多的收益。

| 模型/请求 | S1 tok/s | S2 speedup/效率 | S4 speedup/效率 | S8 speedup/效率 |
|---|---:|---:|---:|---:|
| Qwen short | 296.05 | 1.469× / 73.5% | 2.568× / 64.2% | 4.323× / 54.0% |
| Qwen long | 149.76 | 1.831× / 91.5% | 2.771× / 69.3% | 3.216× / 40.2% |
| DeepSeek short | 180.03 | 1.589× / 79.5% | 2.664× / 66.6% | 4.688× / 58.6% |
| DeepSeek long | 87.93 | 1.822× / 91.1% | 2.723× / 68.1% | 3.137× / 39.2% |

短请求到S8仍有4.3×–4.7×吞吐；长请求从S4增加到S8只再快约15%–16%。slot越多，收益越小。

## 4. KV Cache 为什么越来越浪费

| 模型/请求 | S1 KV | S2 KV | S4 KV | S8 KV | S8有效比例 |
|---|---:|---:|---:|---:|---:|
| Qwen short | 0.938 MiB | 1.875 MiB | 3.750 MiB | 7.500 MiB | 46.25% |
| Qwen long | 24.188 MiB | 48.375 MiB | 96.750 MiB | 193.500 MiB | 46.85% |
| DeepSeek short | 2.188 MiB | 4.375 MiB | 8.750 MiB | 17.500 MiB | 46.25% |
| DeepSeek long | 56.438 MiB | 112.875 MiB | 225.750 MiB | 451.500 MiB | 46.85% |

当前实现按“slot数 × 本组最大长度”预留。S8有8张同样大的草稿纸，但并非8条请求总在同一时刻
达到最大长度，所以预留量翻倍，有效比例反而降到约47%。长请求S8吞吐收益小、Cache代价大，
说明以后应研究分尺寸Cache页或动态块，而不是继续盲目增加slot。

## 5. 答案是否完全相同

- Qwen short/long：所有slot完全一致；
- DeepSeek long：所有slot完全一致；
- DeepSeek short：S1/S2一致，S4/S8一致，但两组彼此不一致；
- 分叉只出现在第6条请求，从第5个生成token开始。

因此执行门是48/48通过，精度总门仍是失败。后续
[top-2诊断](continuous-divergence.zh-CN.md)已经证明prefill B1/B2是因果变量，S4/S8的margin只有
0.000669；默认B2在该请求反而与PyTorch一致，因此不能简单回退成串行prefill。

![Fixed-request slot sweep](../optimization-log/assets/continuous-slot-sweep.svg)

脚本使用`--suite slot-sweep`。完整原始数据见
[Experiment 103](../optimization-log/experiments/103-fixed-request-slot-sweep.md)。
