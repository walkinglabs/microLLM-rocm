# 2026-08-25 — current cached-decode profile runner

## 解决什么证据问题

直接profile一次完整程序会把权重load、Kernel lazy setup、warm-up、cache prepare和decode混在一起。
新runner对同一二进制运行两次：

```text
load + 1次完整generation
load + 3次完整generation
```

两个进程都使用相同warm-up。Kernel统计做`(three - one) / 2`，得到一份完整T2048/B2/N64
generation的平均Kernel成本，而不是把load算成decode。

## 固定输出

`profile_current_cached_decode.py`保存：

- 1-step与3-step应用JSON；
- Kernel stats；
- HIP API stats；
- memory copy stats；
- memory allocation stats；
- phase-delta Kernel分类和top 30；
- 最终summary/raw JSONL。

命令固定BF16 cache、精确`context + new_tokens`容量、device argmax、一次token一次forward，并检查
measured forward数等于`batch × decode_tokens × steps`。

`profile_step_delta.py`新增`cached Attention`与`KV cache store`分类，但不改变旧training/prefill
结果。新静态合同由CTest注册，测试文件审计为127。

runner先推送再测量；任何正式trace必须来自干净revision。
