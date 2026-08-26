# Pinned official Hugging Face fixtures

The fixture tool inspected the real local Qwen2.5-0.5B and
DeepSeek-R1-Distill-Qwen-1.5B files at their pinned revisions. It parsed the complete
safetensors headers, verified every required tokenizer/config file, and reproduced the
expected parameter and Tensor counts.

| model | license | tensors | parameters | weight bytes | tokenizer files |
|---|---|---:|---:|---:|---|
| Qwen2.5-0.5B | Apache-2.0 | 290 | 494,032,768 | 988,097,824 | vocab + merges |
| DeepSeek-Distill-1.5B | MIT | 339 | 1,777,088,000 | 3,554,214,621 | vocab + merges |

Both weight sets contain BF16 payloads and pass the `fixture-ready` contract. Model and
tokenizer payloads remain outside Git; `summary.json` contains only pinned source,
revision, license, counts and byte evidence.
