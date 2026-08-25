# Experiment 234 — SwiGLU算子快两成，整模为什么只动千分之五

Status: `keep explicit operator; reject Auto route`

## 先把错的尺子换掉

profile中BF16 SwiGLU是最大的未关闭非Attention kernel：Qwen/DeepSeek每步约
0.373/0.714 ms。第一版通用benchmark每轮分配输出Tensor，Event测到约0.15 ms，
与profile的0.015–0.025 ms不是同一个问题。这组数字被作废。

正式门新增调用者提供输出的`swiglu_out_with_implementation_`，scalar和vectorized反复
写同一块Storage，只测kernel。vectorized每个线程处理4个BF16值，tail另走安全循环。

## Operator gate

| Shape | Elements | Scalar mean median | Vector mean median | Speedup | Complete output |
|---|---:|---:|---:|---:|---|
| Qwen B1T1024 | 4,980,736 | 0.01707 ms | 0.01367 ms | 1.249× | bit-identical |
| DeepSeek B1T1024 | 9,175,040 | 0.02791 ms | 0.02345 ms | 1.190× | bit-identical |

每格3个fresh processes、3 warm-up、30 Event calls。operator门通过，进入整模。

## Full-model gate

使用当前grouped FFN/BTHD B1T1024路径，baseline/candidate交替顺序，每格3进程、2 warm-up +
5 measured，比较完整vocab logits。

| Model | Speedup | Logit Max/RMS | Peak/allocation | Result |
|---|---:|---:|---|---|
| Qwen | 1.0073× | 0 / 0 | unchanged | pass 1.005× |
| DeepSeek | 1.0005× | 0 / 0 | unchanged | performance fail |

![BF16 SwiGLU vector discard](../assets/bf16-swiglu-vector-discard.svg)

## 决定

- 保留`SwiGLUImplementation::Vectorized`和caller-output API；
- `Auto`继续走scalar，官方模型默认不变；
- 不用Qwen的0.73%掩盖DeepSeek只有0.05%；
- 这条micro-kernel路线关闭。要继续减少FFN时间，必须跨过SwiGLU边界，例如与
  grouped GEMM epilogue融合，而不是再调每线程元素数。

发布回归：CPU 344/344、ASan/UBSan 342/342、PyTorch-enabled 318/318、完整CPU/HIP
542/542（3个条件跳过）、HIP标签186/186。覆盖清单注册106个测试文件。

证据：

- [`operator gate`](../../../benchmarks/results/2026-08-25-bf16-swiglu-vector-operator/)
- [`full-model gate`](../../../benchmarks/results/2026-08-25-bf16-swiglu-vector-model-gate/)
