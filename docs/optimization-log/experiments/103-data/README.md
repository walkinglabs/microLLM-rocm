# Experiment 103 data

- `before-fix-raw.jsonl`: 48 attempted processes; 30 pass and 18 stable refill failures.
- `after-raw.jsonl`: the unchanged 48-process matrix after the full-row recycle fix; 48 pass.
- `after-summary.json`: min/p50/max, S1 speedup/efficiency, exact KV/peak and token differences.
- `environment.txt`: fixed MI300X and workload protocol.
- `gates.json`: repository and focused gates.

The summary intentionally has passing execution and a recorded accuracy failure: DeepSeek short
changes one request between S1/S2 and S4/S8.
