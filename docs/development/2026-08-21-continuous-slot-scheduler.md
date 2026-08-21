# Continuous slot scheduler reference

`ContinuousBatchScheduler` now owns a fixed-row shared `KVCache`, admits pending requests into
free rows, resets completed/cancelled rows, and refills them on later steps. It supports independent
prompt lengths, generation limits, stop tokens, RNG state and FP32/BF16 cache policies. Greedy HIP
selection reduces all slot logits together and copies one small `[slots]` result per scheduler step.

CPU tests align length refill and delayed stochastic sampling against independent generation and
exercise stop/cancel/policy failures. HIP tests align FP32/BF16 output and memory metrics with CPU.
The benchmark reports slot utilization, refills, logical/dummy rows, uniform/divergent calls and
allocated/active cache bytes.

MI300X Release tiny-model throughput is 0.748x–0.858x the serial reference because every measured
batch call is divergent and therefore executes the serial B1 model oracle. This retained negative
result defines the baseline for positions-aware parallel kernels.

A Release uniform control reaches 1.434x/1.904x/2.356x the serial reference at B2/B4/B8, proving
that the shared batch path can scale. It is still only 0.680x/0.488x/0.308x the static batch because prompt
prefill remains row-serial and the scheduler retains per-step state/selection work.

See [Experiment 096](../optimization-log/experiments/096-continuous-slot-scheduler.md) and the
[beginner guide](../dev/continuous-slot-scheduler.zh-CN.md).
