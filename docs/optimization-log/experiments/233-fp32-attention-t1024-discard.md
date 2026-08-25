# Experiment 233 — 四个局部winner，为什么没有一个能默认启用

Status: `reject T1024 exact Attention policy`

## 从profile到四个精确问题

Experiment 232把当前B1T1024的GEMM占比定位为59.7%/66.8%。本轮不扫描所有模型GEMM，只筛
Attention QK/PV四个描述符。每格3个fresh processes、最多64个算法；所有candidate必须先比较
完整输出，再做2 warm-up +5 Event。

| Shape | Winner | Operator speedup | Max/RMS |
|---|---:|---:|---:|
| Qwen QK | 304680 | 1.538× | 8.94e-8 / 1.61e-8 |
| Qwen PV | 294867 | 1.476× | 2.46e-7 / 5.09e-8 |
| DeepSeek QK | 310758 | 1.060× | 0 / 0 |
| DeepSeek PV | 296917 | 1.103× | 0 / 0 |

四格各有64个三进程共同正确candidate，局部门全部过1.05。

## 第一个反例：PV名字相同，descriptor不同

当前BTHD模型的PV消费`V[B,T,H,D]`交错布局，使用专用hipBLASLt descriptor；standalone tuner筛的
是普通BHTD descriptor。把Qwen PV index注册到模型后，CLI报告1个entry、175次miss、0 hit、
0 dispatch。runner在计时前停止，未生成伪PV收益。

## 第二个反例：QK命中了，也不能默认

正式模型门只保留真正可命中的QK：B1T1024、当前grouped/BTHD策略、baseline/QK、每格3进程。

| Model | Exact hits | Model speedup | Complete-logit Max/RMS | Result |
|---|---:|---:|---:|---|
| Qwen | 168 | 1.051× | 0.0733 / 0.0157 | correctness fail |
| DeepSeek | 196 | 1.002× | 0 / 0 | performance fail |

![T1024 Attention solutions discard](../assets/fp32-attention-t1024-discard.svg)

Qwen局部winner确实带来约5.1%整模收益，但不同合法累加顺序经过24层后放大，失败1e-4/1e-5门。
DeepSeek完全对齐，却只有0.2%收益。不能用两个模型各过一半拼成一个政策。

## 决定

- 不注册任何T1024默认index；
- QK/PV operator结果和PV descriptor mismatch全部保留；
- 若未来为interleaved BTHD PV建立专用tuner，必须使用真实descriptor重新筛；
- exact solution模型track再次关闭，下一方向必须重新选择，不降低完整logits门。

证据目录：

- [`T1024 operator screening`](../../../benchmarks/results/2026-08-25-fp32-attention-t1024-solutions/)
- [`BTHD PV mismatch pilot`](../../../benchmarks/results/2026-08-25-fp32-attention-t1024-bthd-pv-mismatch-pilot/)
- [`QK full-model gate`](../../../benchmarks/results/2026-08-25-fp32-attention-t1024-qk-model-gate/)
