# Experiment 095 data map

- `pilot-raw.jsonl`: 24 fresh-process framework records for T1/32/128, B2/4, N64.
- `pilot-summary.json`: 12 paired case rows, including one preserved microLLM failure.
- `long-raw.jsonl`: four fresh-process framework records for T2048, B2, N64.
- `long-summary.json`: two successful long-context paired rows.
- `qwen-t128-b4-n64-recheck/`: three independent microLLM reruns of the pilot failure shape.
- `summary.json`: compact combined findings and interpretation boundaries.
- `gates.json`: executable and evidence counts.
- `environment.txt`: frozen artifact, device and measurement contract.

The initial Qwen failure is intentionally retained even though three immediate fresh-process
reruns passed. It is an observed non-stable failure, not a proven stable indexing bug.
