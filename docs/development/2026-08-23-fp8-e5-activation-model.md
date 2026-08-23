# E5M2 activation：底层支持保留，模型策略拒绝

最初节点给模型增加E5M2-FNUZ activation，同时Linear权重保持E4M3-FNUZ。历史命令如下，
当前CLI已经不再接受它：

```text
--fp8-linear true
--fp8-activation-format e5m2-fnuz
```

这个接口只为完成同revision反驳实验存在，不是已经证明可用的公开策略。

Autograd now accepts independent left/right FP8 dtypes. E5 activation/E4 weight forward uses the
same mixed operand pair as graph-free inference; gradients remain FP32 straight-through updates to
FP32 masters. Existing calls default to E4/E4.

CPU与HIP gate证明混合格式可以执行：Tensor dtype、量化/反量化、独立左右operand dtype的
autograd以及MI300原生E5×E4 GEMM都成立。这些只能证明原语存在。

Exp153随后用Qwen/DeepSeek、T8/T512、各三次独立进程比较同binary的E5与E4。E5八项完整logits
Max/RMS全部恶化：比例为1.508×–3.428×；两项T512速度门通过、显存无变化，但完整precision
仍为0/4。

结论：删除`Fp8ActivationFormat`、模型字段、CLI参数和通用matrix参数。保留底层E5 dtype、
量化算子、混合operand autograd API与原生GEMM测试，供未来其他模型显式研究。历史命令只在
[Experiment 153](../optimization-log/experiments/153-fp8-e5-activation-discard.md)中复现，
不能作为当前接口使用。

## 删除策略后的回归

| Gate | 结果 |
|---|---:|
| CPU Debug | 251/251 |
| ASan/UBSan | 249/249 |
| PyTorch-enabled CPU | 225/225 |
| 完整CPU/HIP配置 | 368/368，2个条件跳过 |
| MI300X HIP标签 | 113/113 |
| optimization log validator | 72 workers / 24 FP8 rows / 0 precision pass |

底层保留门在最终MI300回归中仍可见：CPU混合格式参考、autograd FP32梯度以及原生
`MixedE5ActivationAndE4WeightExecuteWithExplicitDispatch`全部通过。CPU coverage重新测得
6,582/7,957 lines、706/779 functions和6,347/9,961 branches。
