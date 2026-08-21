# DeepSeek slot divergence and prefill counterfactual

Continuous scheduling now has an opt-in selection diagnostic. It records source path, actual logit
batch, request/slot/position, device argmax, top-1/top-2 logits and margin. The default is off because
diagnostic host copies invalidate performance timing.

Eighteen DeepSeek short processes locate the first S1/S2 versus S4/S8 divergence at request 5,
generated index 4. The S4/S8 margin is 0.000669 and both paths swap tokens 23606/1196. Device argmax
always matches the host top-1 check.

An experimental `batch_equal_length_prefill=false` control preserves positions-aware B4/B8 decode
while serializing only prompt admission. Both counterfactual cases return to the complete S1 output
and reproduce S2 logits, isolating batched prefill as the causal variable. The control is not the
default: at the original divergence, default B2 matches the sequential PyTorch full-BF16 token while
serial B1 adds another external mismatch.

See the [beginner explanation](../dev/continuous-divergence.zh-CN.md) and
[Experiment 104](../optimization-log/experiments/104-deepseek-prefill-divergence.md).
