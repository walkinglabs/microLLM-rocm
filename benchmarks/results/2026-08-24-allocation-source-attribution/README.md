# Allocation source×size attribution

Experiment 186 adds opt-in thread-local allocation diagnostics and tags graph-free
inference regions. It profiles one T512 B1 prefill under the retained BF16 FFN
`minimum_rows=512` policy; rejected QKV Arena is off.

Three fresh processes per model produce identical source×size distributions:

| Model | Calls | Logical allocated bytes | Attention core |
|---|---:|---:|---:|
| Qwen | 580 | 1,079,854,592 | 144 calls / 572,522,496 bytes |
| DeepSeek | 676 | 1,817,003,520 | 168 calls / 792,723,456 bytes |

For Qwen, the 14,680,064-byte score/probability-class allocation appears 24 times;
for DeepSeek, 12,582,912 bytes appears 28 times. The second core family is the hidden
width allocation: 1,835,008×120 and 3,145,728×140.

`attention.core` is the common top source. It is 53.0%/43.6% of logical allocated
bytes for Qwen/DeepSeek. The next optimization must start there rather than moving
another guessed projection/FFN Tensor.

Files: `raw.jsonl`, `summary.json`, and `verification.json`.
