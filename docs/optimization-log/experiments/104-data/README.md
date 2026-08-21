# Experiment 104 data

- `raw.jsonl`: six DeepSeek short cases × three fresh processes = 18 diagnostic records.
- `summary.json`: first divergence, top-2/margin/source/batch evidence, serial-prefill
  counterfactual and PyTorch token comparison.
- `environment.txt`: device and diagnostic boundary.
- `gates.json`: focused and repository gates.

Diagnostics copy logits to the host and are not performance measurements. Default S4/S8 use
batched prefill; the two `_serial_prefill` cases change only prompt admission batching.
