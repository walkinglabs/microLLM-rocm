# Device token history after allocator stabilization

Experiment 090 adds caller-owned argmax outputs and a device-resident token history for HIP greedy
generation without stop tokens. The next-token Tensor remains on device; one history transfer at
completion replaces one D2H transfer per token. Sampling and early-stop paths are unchanged.

DeepSeek T2048 B1/B8 alternating Release medians are 1.002x/1.003x. Qwen T512 B8 is 0.997x.
D2H calls fall 24 to 3 with unchanged bytes, allocator counters remain stable, and peak/KV/token
contracts pass. Public B1/B3 generation tests verify one final D2H.

See [Experiment 090](../optimization-log/experiments/090-device-token-history.md) and the
[beginner guide](../dev/device-token-history.zh-CN.md).
