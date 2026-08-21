# Experiment 072 evidence

- `raw.jsonl`: CPU/HIP × 1/2/4/8 requests × three fresh processes.
- `summary.json`: median throughput, serial/sequential ratio, active requests and Cache bytes.
- `build-contract.json`: fixed tiny serving workload.

This is a correctness/reference baseline. It performs no cross-request batched forward and must not
be presented as a production serving speedup.
