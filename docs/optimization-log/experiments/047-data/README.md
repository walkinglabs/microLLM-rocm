# Experiment 047 evidence

- `candidate/raw.jsonl`: matched Experiment 044 protocol, three fresh microLLM/PyTorch pairs.
- `candidate/summary.json`: generated medians for one Qwen `1×128` shape.
- `protocol-mismatch/`: the first `2 warm-up + 5 measured` run. It is retained because
  comparing it to the old `1+2` baseline would have been invalid.
- `comparison.json`: matched-protocol baseline/candidate calculation and discard decision.

The source candidate is intentionally absent: it failed the throughput gate and was
removed before this evidence-only commit.
