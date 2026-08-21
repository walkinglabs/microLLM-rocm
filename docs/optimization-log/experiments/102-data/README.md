# Experiment 102 data

- `micro-raw.jsonl`: 2 models × 4 cases × 3 fresh processes = 24 complete records.
- `micro-summary.json`: min/p50/max aggregates plus the full raw rows.
- `pytorch/`: eight sequential-request PyTorch BF16 reference JSON records.
- `comparison.json`: generated exact-token, throughput-boundary, KV and memory join.
- `environment.txt`: frozen device, runtime and measurement protocol.
- `gates.json`: final repository test counts and recorded accuracy status.

The PyTorch program is intentionally sequential. Its throughput is not a matched scheduler oracle.
Generated token arrays are the correctness oracle: Qwen passes 4/4 and DeepSeek passes 1/4.
