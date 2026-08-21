# Experiment 073 evidence

- `raw.jsonl`: CPU/HIP × B1/B2/B4/B8 × three fresh processes.
- `summary.json`: static-batch throughput, reference speedup, scaling efficiency and memory.
- `build-contract.json`: compatible-request workload and model.

Static batching requires equal prompt length and one shared generation configuration. It is not yet a
continuous scheduler for delayed or uneven requests.
