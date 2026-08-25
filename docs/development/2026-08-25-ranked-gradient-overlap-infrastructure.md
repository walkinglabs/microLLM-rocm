# 2026-08-25 — ranked gradient-ready overlap infrastructure

Experiment 269建立了one-process-per-GPU persistent bucket views。本节点只改变通信enqueue时机，
不改变Storage、bucket范围、collective数量、view、optimizer或数值门。

## Lifecycle and ordering

第1步同步建立plan。从第2步开始：

```text
optimizer.zero_grad
-> begin_overlap_step
-> backward leaf gradient-ready hook
-> default Stream records one Event when a bucket is complete
-> communication Stream waits Event
-> pack + RCCL average in fixed bucket 2 -> 1 -> 0 order
-> finish_overlap_step waits the tail
-> optimizer.step
```

即使较低range先ready，plan也只能按固定逆序enqueue，保证两个独立rank进入相同RCCL序列。
重复ready、缺失parameter、未初始化/non-view plan、提前finish和重复begin都会报错。

active plan在异常析构/clear时先abort communicator，再best-effort device synchronize，最后释放
Storage。测试注入只完成部分bucket的状态，确认clear后plan inactive且communicator aborted，避免
异步工作继续引用已释放Storage。

## Pilot

Model-S、两rank、三步、25 MiB真实双进程：

- overlap flags `[0,1,1]`，overlapped buckets `[0,3,3]`；
- finish wait约1.326/1.304ms；同步views pilot约2.8ms；
- later backend allocation为0，pack 57、unpack 0；
- current/peak精确等于同步views的249,378,816/324,929,288 bytes；
- plan容量62,344,704 bytes，完整参数/CPU/loss/故障门通过；
- 五策略matrix pilot的finish wait约1.998×，但完整step相对同步views仅0.985×。

因此正式结果可能拒绝overlap，不能从finish wait单独得出加速。完整RCCL标签47/47，ranked
contract 5/5，测试文件仍123。下一提交从干净revision跑五策略各三次。
