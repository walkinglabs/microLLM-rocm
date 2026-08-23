# Zero-stride GQA Value broadcast evidence

Experiment 169 asks hipBLASLt to broadcast one V head across a query-head group with
matrix batch stride zero. It avoids the expanded Value Tensor and writes context BTHD.

- `raw.jsonl`: five shapes × repeated/broadcast × three fresh processes;
- `summary.json`: Event/wall medians, complete-output gates and counterexamples;
- `coverage-summary.json`: post-change CPU coverage;
- `verification.json`: capability decision and regression gates.

All 30 rows pass complete-output and zero-transfer gates. Results split by width/model:
Qwen T128/T512 is slower (`0.946×/0.937×` wall), while DeepSeek T512 is `1.603×`.
The MHA repeats=1 counterexample is `0.726×`. The universal route is rejected; the explicit
primitive remains because the width-128 result justifies a selective full-backward
experiment. No model default changes in this node.
