# Experiment 085 data map

- `qwen-raw.jsonl`, `deepseek-raw.jsonl`: frozen 72-record output-length semantic/KV/memory
  survey. Its build type was unspecified, so do not publish its throughput.
- `summary.json`: machine summary of that 36-shape semantic survey.
- `qwen-release-raw.jsonl`, `deepseek-release-raw.jsonl`: frozen Release/gfx942 N8
  performance records, 24 process rows.
- `release-summary.json`: 12 paired Release shapes and microLLM/PyTorch ratios.
- `invalid-free-first-token-pilot.jsonl`: the pilot that counted the token already produced by
  prefill without running a cached model step.
- `mixed-qwen-runner-invalid.jsonl`, `mixed-deepseek-runner-invalid.jsonl`: runs invalidated
  because the shared runner/binary changed during collection.
- `environment.txt`: exact runtime, build-type boundary and frozen artifact paths.
- `gates.json`: CPU, HIP, sanitizer and schema-test counts.

The invalid files are evidence of rejected procedures, not performance baselines.
