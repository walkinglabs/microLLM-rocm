# Experiment 099 — 先把trace洗干净，再拒绝一次scatter

普通scheduler benchmark把五条实现放在一个进程，第一次trace里matmul占约64%，但权重copy、serial和
sequential也混在里面。这个证据不能回答continuous自身的问题。

## 新的profile模式

`--continuous-only true`只构造一个模型，只运行continuous warmup和measurement，输出精确H2D/D2H/
D2D与allocation counter。JSON明确写`correctness_gate=external_full_suite`，不伪造进程内对照。

![Continuous-only profile](../assets/continuous-profile-scatter-discard.svg)

## 干净trace

| 类别 | R8/S4 | R8/S2 |
|---|---:|---:|
| wall | 12.992 ms | 19.774 ms |
| Kernel | 8.766 ms / 1679 calls | 12.549 ms / 2287 calls |
| typed GEMM | 61.90% | 62.89% |
| copyBuffer | 9.28% | 9.26% |
| positioned三Kernel | 5.84% | 7.84% |
| H2D | 32 calls / 596 B | 56 calls / 596 B |
| D2H | 9 / 144 B | 17 / 136 B |
| D2D | 159 / 113,664 B | 159 / 113,664 B |

同步`hipMemcpy`的API duration会吸收前序GPU工作，不能解释成实际复制时间。Kernel trace和引擎
counter必须一起读。

## 假设

active logits目前逐row D2D回填固定slot。假设一次GPU scatter能减少小copy启动，吃掉copyBuffer
热点的一部分。候选有CPU/HIP row映射测试，scheduler每compacted step调用一次，checksum正确。

## 反驳结果

| shape | baseline median | scatter median | candidate/baseline | normalized old→new |
|---|---:|---:|---:|---:|
| R8/S4 | 3938.55 | 3910.60 | 0.993× | 1.6448→1.6172 |
| R8/S2 | 2772.30 | 2697.71 | 0.973× | 1.1582→1.1278 |

scatter需要额外上传row mapping并启动compute Kernel，原来约1KiB的copy engine路径已经很便宜。
更重要的是，159次D2D不全来自logits回填，还包含prefill Cache copy和materialization。假设把整个
copyBuffer占比错误归因给一个调用点，因此候选完整回退。

## 保留什么

保留continuous-only profile模式、schema smoke、两份pftrace和失败数据；不保留scatter公共op、HIP
Kernel、scheduler route或指标。下一步应尝试合并token/position/row三份小metadata上传，或者对官方
模型服务路径profile，而不是重新实现同一个scatter。

数据见 [`099-data`](099-data/)。
