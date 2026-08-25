# Scoped Autograd gradient-producer matrix

Experiment 261 tests the caller-owned producer inside repeated backward on an
already-built graph. It uses the same four Model-S shapes and tiny counterexample,
three fresh processes per shape, 5 warm-ups, 40 repetitions, and alternating
baseline/direct plus shape order.

| Shape | Event speedup | Wall speedup | Decision |
|---|---:|---:|---|
| Model-S head T32 | 0.993x | 0.994x | reject |
| Model-S FFN T32 | 0.995x | 0.999x | reject |
| Model-S Attention T32 | 0.976x | 0.991x | reject |
| Model-S head T512 | 1.035x | 1.018x | reject |
| tiny counterexample | 1.001x | 1.005x | reject |

Every complete gradient is exact, every caller address is preserved, and one
logical allocation disappears per invocation. No shape clears the 1.05 Event
and wall gate. Ordinary first leaf assignment already adopts the producer Tensor
without an add; the scoped state machine therefore has no measured benefit.

The Autograd route and target-state API are removed next. The independently fast
caller-owned operator remains public for explicit workspaces and future fused
producers.
