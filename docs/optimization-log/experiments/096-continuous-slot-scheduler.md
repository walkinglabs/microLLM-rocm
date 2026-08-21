# Experiment 096 — 真正补位跑通了，但第一版更慢

Experiment 094只证明模型能把新prompt写入一个空row。本节点把submit、pending、slot、stop、cancel、
reset、row prefill和divergent decode接成一个完整状态机。

## 任务合同

固定`max_slots`行共享KV Storage。每个scheduler step最多让每个已占用slot输出一个token；完成的row
立即reset，pending请求在下一个step按最低空slot补入。必须逐请求等于独立B1生成，并同时报告真实
slot、Cache与dummy计算。

## 状态变化

```text
step 1  [A,B] → 两行输出 → positions [4,4]
step 2  A完成，B继续       → [0,5]
step 3  C补入row 0，B继续  → [3,6]
step 4  C和B完成           → [0,0]
```

greedy HIP选择先对`[slots,1,V]`做一次row-wise argmax，再只复制`[slots]`，所以D2H calls等于
scheduler steps，而不是active requests之和。

![Continuous slot scheduler](../assets/continuous-slot-scheduler.svg)

## 正确性证据

- A、B、C逐请求等于独立B1；
- FP32/BF16 Cache均通过；
- 延迟随机采样保持每请求独立seed；
- length、stop和cancel都释放row；
- C复用A的slot 0，B的输出不变；
- shared allocated Cache在全部完成后仍可复用，active Cache回到0；
- HIP输出与CPU一致，greedy每step一次小D2H；
- KV策略、slot数量、ID和最大step错误明确失败。

## MI300X性能反例

先用未指定build type的诊断binary确认计数；随后以完全相同shape构建Release，执行2次warmup和10次
measured repetition。下面是主要Release结果：

| requests | slots | continuous/reference | slot利用率 | refill | divergent calls | dummy rows |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 0.858× | 87.5% | 0 | 3 | 1 |
| 4 | 2 | 0.824× | 93.8% | 2 | 7 | 3 |
| 8 | 2 | 0.781× | 91.2% | 6 | 16 | 9 |
| 4 | 4 | 0.748× | 75.0% | 0 | 4 | 5 |
| 8 | 4 | 0.794× | 86.1% | 4 | 8 | 9 |

5/5输出与sequential完全一致，但全部更慢。所有batch decode都是divergent calls，uniform calls为0；
当前模型因此逐row执行完整B1，还要为固定batch计算dummy row。4槽预留128KiB，实测active峰值只有
24–24.5KiB，说明小队列还存在明显容量空洞。

为了避免把“状态机必然慢”当作解释，又加了等长、等生成长度、slots=requests的uniform对照：

| requests/slots | continuous/reference | continuous/static | uniform | divergent | dummy |
|---:|---:|---:|---:|---:|---:|
| 2 | 1.434× | 0.680× | 3 | 0 | 0 |
| 4 | 1.904× | 0.488× | 3 | 0 | 0 |
| 8 | 2.356× | 0.308× | 3 | 0 | 0 |

uniform路径确实超过串行reference，证明共享batch计算有效；但仍远慢于一次完整static batch，且B8
只有static的30.8%。原因是每个请求仍逐row prefill，scheduler每token还做状态管理与小D2H。
因此下一步既要消除divergent串行，也要把兼容prompt的prefill合批。

## 接受什么，拒绝什么

保留状态机和公共API，因为它第一次跑通真实补位，且为并行Kernel提供稳定oracle。拒绝“continuous
已经加速”的解释：当前证据直接反驳它。

下一节点只改变计算层：让RoPE、K/V store和cached Attention直接读取`positions[B]`与active mask，
一次Kernel路径处理所有真实row，并停止dummy row推进。只有逐rowlogits、请求结果和状态转移继续
通过本节点门，才能比较吞吐。

原始JSON见 [`096-data`](096-data/)。
