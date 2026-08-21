# Experiment 072 — 先让多请求正确，再谈continuous batching

## 旧边界

现有batched KV要求batch内所有请求共享长度和position。真实服务的请求会晚到、早结束、使用
不同prompt和随机种子。直接改成可变slot Kernel之前，需要逐请求金标准。

## 实现

`ReferenceScheduler`维护独立B=1 KV Cache和状态机：

```text
PendingPrefill → Decoding → Completed
```

每个scheduler step让所有活跃请求各生成一个token；中途可`submit()`新请求。完成时立即释放
Cache。公开snapshot和metrics记录arrival/completion step、prefill/decode calls、活跃请求和
Cache峰值。

CPU测试包含独立随机数流和延迟到达，并与两个独立`generate()`逐token对齐；HIP测试与CPU
reference对齐。零token请求立即完成，非法prompt/context/policy和step上限都有显式错误。

## 串行基线

106,816参数tiny模型，prompt 4/8/12/16，生成3/4/5 token，CPU/HIP、1/2/4/8请求，1次
warm-up、5次measured、3个新进程中位数：

| device | requests | scheduler tok/s | sequential tok/s | ratio | peak Cache |
|---|---:|---:|---:|---:|---:|
| CPU | 1 | 367.1 | 363.6 | 0.994× | 3,584B |
| CPU | 2 | 346.4 | 346.4 | 0.998× | 9,728B |
| CPU | 4 | 306.8 | 317.9 | 0.961× | 28,160B |
| CPU | 8 | 306.1 | 312.4 | 0.980× | 56,832B |
| HIP | 1 | 330.9 | 333.7 | 0.992× | 3,584B |
| HIP | 2 | 331.4 | 334.3 | 0.992× | 9,728B |
| HIP | 4 | 330.5 | 333.1 | 0.993× | 28,160B |
| HIP | 8 | 331.3 | 333.1 | 0.994× | 56,832B |

![Reference serving scheduler](../assets/reference-serving-scheduler.svg)

所有输出相同。HIP吞吐随请求数几乎不变，因为仍是B=1串行forward；这不是失败的优化，而是
下一节点的明确before：batched forward calls为0。

## 决定

保留reference scheduler、benchmark和指标API。不能宣传continuous batching已经完成。
下一候选必须把可兼容请求合成slot batch，并同时通过延迟到达、不同长度、随机状态、完成
顺序和Cache释放合同。
