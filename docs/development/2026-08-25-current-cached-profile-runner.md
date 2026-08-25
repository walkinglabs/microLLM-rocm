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

第一次干净执行在应用合同处被runner拒绝。原因不是模型失败，而是原始`hf_infer`记录使用
`token_count`，`model/context/decode_tokens`由framework runner归一化后才存在；profile runner
错误地要求原始记录已有后三个字段。修复后改为检查`token_count/context`和完整forward计数，再
显式写入profile元数据。错误信息也会列出每个actual/expected字段，不再只说“contract changed”。

这次失败发生在任何trace结论生成前。修复必须单独推送，正式profile从修复提交重新开始。
