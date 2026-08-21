# Continuous-only profile：只看调度器自己花了多少时间

## 1. 为什么普通benchmark不能直接profile

普通scheduler benchmark会依次运行serial reference、sequential generate、static batch和continuous。
如果直接用rocprof包住整个进程，看到的Kernel总表把所有方法混在一起，甚至模型权重初始化的大copy
也会算进HIP API时间。

新模式：

```bash
microllm_bench_scheduler \
  --device hip \
  --requests 8 \
  --continuous-slots 4 \
  --continuous-only true \
  --warmup 1 \
  --repetitions 1
```

它只热身和测量continuous。输出不假装做了进程内reference比较，而是写：

```text
correctness_gate = external_full_suite
```

正确性由完整CPU/HIP测试和前一节点的交替A/B负责。

## 2. 新时间线看见什么

| 类别 | R8/S4 | R8/S2 |
|---|---:|---:|
| Kernel总时间 | 8.766 ms | 12.549 ms |
| typed GEMM | 61.9% | 62.9% |
| copyBuffer | 9.28% | 9.26% |
| positioned RoPE/store/Attention | 5.84% | 7.84% |
| H2D | 32 calls / 596 B | 56 calls / 596 B |
| D2H | 9 calls / 144 B | 17 calls / 136 B |
| D2D | 159 calls / 113,664 B | 159 calls / 113,664 B |

HIP API里同步`hipMemcpy`可能吸收前面尚未完成的GPU工作，所以不能把API duration直接说成复制
带宽时间。真实方向、calls和bytes以引擎counter为准。

## 3. 一个看起来合理但失败的优化

我们猜：逐active-row把logits D2D复制回固定slot造成许多小copy，改成一次GPU scatter应该更快。
Kernel和测试都正确，但三对交替A/B结果：

- R8/S4：0.993×，下降0.71%；
- R8/S2：0.973×，下降2.69%。

新scatter还要上传row mapping并启动compute Kernel，原来约1KiB的copy engine路径已经足够便宜。
所以候选完整回退，只保留失败数据。

## 4. 下一步怎样选

不能继续把整个9.3% copyBuffer都归罪于logits scatter。它还包含row-prefill Cache复制、Tensor
materialization等来源。下一步可以测把token/position/row三份小H2D合并成一份metadata，或为真实
Qwen/DeepSeek做profile；仍需先定合同再改。

实验记录见 [Experiment 099](../optimization-log/experiments/099-continuous-profile-scatter-discard.md)。
