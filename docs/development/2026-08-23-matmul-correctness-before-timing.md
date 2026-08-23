# Matmul自动候选：先答对，再计时

## 为什么“跑得快”不能排第一

如果一个候选给出错误矩阵，它可以因为少算了一部分而特别快。正确流程必须是：

```text
Readable完整reference
→ 每个候选完整输出Max/RMS/finite
→ 失败候选停止，时间保持0
→ 通过候选warm-up
→ HIP Event与wall重复计时
→ P50/P95排序
→ 只返回推荐
→ 模型端到端回归后，调用者显式接受
```

`autotune_matmul()`不会修改registry。只有`register_matmul_autotune_winner()`会注册，而且会再次
确认被推荐项具有supported、finite、correctness和正Event计时证据。

## API

```cpp
microllm::ops::MatmulAutotuneOptions options;
options.warmup = 3;
options.repetitions = 10;
options.mode = microllm::ops::OpMode::Inference;

const auto report = microllm::ops::autotune_matmul(
    left, right, false, false, options);

// 先运行模型端到端回归，再显式接受：
microllm::ops::register_matmul_autotune_winner(report);
microllm::ops::save_matmul_tuning_cache("matmul-cache.jsonl");
```

Autotune使用legacy default Stream的Event，不创建non-default Stream，因此不会永久关闭exact-size
allocator。dtype默认误差门可被显式覆盖；0表示要求bit-exact。

## CLI

```bash
./build/hip-release/benchmarks/microllm_tune_matmul \
  --m 128 --k 128 --n 128 --dtype fp32 \
  --warmup 3 --repetitions 10 --mode inference \
  --accept false
```

`--accept true --cache-output path.jsonl`是显式接受。输出会保留
`"end-to-end acceptance remains external"`，避免把算子选择写成模型结论。

## 执行反例

MI300上：

- FP32 64³两个候选都正确，完整报告Event/Wall P50/P95；
- FP32 128³推荐hipBLASLt，显式接受后写出一条persistent entry；
- FP16 64³在零容差下，hipBLASLt Max/RMS非零，被拒绝且四项时间均为0；Readable被推荐；
- autotune前后allocator仍enabled。

原始JSON与cache在
[`benchmarks/results/2026-08-23-matmul-correctness-before-timing/`](../../benchmarks/results/2026-08-23-matmul-correctness-before-timing/)。

## 仍然缺什么

当前候选只有Readable与hipBLASLt默认路径，还没有枚举具体solution index；也没有自动运行模型
端到端回归。Attention、Softmax、RMSNorm等其他热点还没有相同registry。下一步应先扩展一个
真实热点，而不是把“自动”理解为无边界搜索。

最终回归：CPU 254/254、ASan/UBSan 252/252、PyTorch-enabled CPU 228/228、完整CPU/HIP
375/375（2个条件跳过），其中HIP标签117/117。
