# Experiment 156：一个少掉的barrier，让同一Attention每次答案不同

精确matmul注册键完成后，完整HIP回归偶发失败。旧revision复跑20进程也有1次token失败，说明
不是注册键引入。把token门改成完整logits后，修复前20进程只有2次过门；固定Q/K/V直接调用
短序列fused causal GQA 20次，每次都与第一次不同，最大差在0.0023–0.0677之间。

trace把首个不同阶段定位到`block0.attention.context`。根因在共用的block reduction：最后一次
`__syncthreads()`发生在读取`scratch[0]`之前。快线程返回后立即开始下一次reduction并覆盖scratch，
慢线程仍在读取上一次Max或Sum。

修复是每个`block_reduce_max/sum/sum_int`先把结果读进寄存器，再增加一次barrier，确认所有lane
已经读取，之后才允许复用scratch。causal GQA继续保持并行，没有换成慢速串行reference。

| Gate | 修复前 | 修复后 |
|---|---:|---:|
| 固定Q/K/V，20次直接Attention | 20/20不同，worst 0.0677 | 20/20 bit-exact |
| T1–128/B1–8/FP32-BF16完整logits，20进程 | 2/20通过 | 20/20通过 |
| 旧revision token矩阵，20进程 | 19/20通过 | 不再用低margin token掩盖logit漂移 |
| Tiny T128/B8 train TPS中位数 | 231,623 | 231,940（+0.14%） |
| 完整CPU/HIP配置 | 1次偶发失败 | 370/370 |

![Block reduction determinism](../assets/block-reduction-determinism.svg)

训练吞吐三进程无回退；旧kernel的三次final loss彼此漂移，修复后三次只剩最后打印位差异。
新的shape gate比较prefill与两步decode的完整logits、finite、Max和RMS，而不是只比较可能恰好
没翻转的argmax token。其他generation测试继续保留严格token一致性。

这是correctness keep，不进入FP32 running-best性能曲线。它同时说明为什么完整回归不能只在失败
后重跑一次：若没有20进程反例和直接算子复现，这个数据竞争会继续以“偶发低margin”隐藏。
