# Official-model continuous serving matrix

The Hugging Face inference CLI now has a `continuous` workload backed by the real
`ContinuousBatchScheduler`. It accepts explicit slot, prompt-length and output-length lists,
runs warm-up outside measurement, emits complete per-request generated tokens, and reports
scheduler, transfer, allocator, resident-weight and KV-cache counters in one JSON record.

The scheduler cache can now be bounded to the maximum sequence required by a known workload.
This changes a two-slot Qwen smoke from a model-maximum allocation of hundreds of MiB to the
exact request-bound formula. Invalid negative, over-model and over-capacity requests fail before
cache writes.

The checked-in MI300X evidence contains 24/24 passing fresh microLLM processes: two official
models, four short/long and 2/4-slot cases, and three processes per case. Complete generated
tokens are stable across all three processes. Qwen matches the sequential PyTorch BF16 reference
in 4/4 cases. DeepSeek matches only `short_s2`; its other three cases are explicit accuracy
failures and block a long-context parity claim.

The measured microLLM/PyTorch quotient compares a continuous slot scheduler with sequential
PyTorch requests and different weight residency policies. It is named an observed service
throughput ratio, never a matched-algorithm speedup.

See the [beginner guide](../dev/official-continuous-serving.zh-CN.md) and
[Experiment 102](../optimization-log/experiments/102-official-continuous-serving.md).
