# Experiment 048 evidence

- `all-tensor-early-stop/raw.jsonl`: one paired Qwen `1×128` process. The `0.577×`
  result triggered the documented early-stop rule.
- `small-tensor/context128/`: three fresh framework pairs for the first corrected shape.
- `small-tensor/rest/`: the other three shapes, again three fresh pairs each.
- `comparison.json`: dispatch counts, Experiment 044 references and discard boundary.

The candidate source and temporary optimizer counters were removed after the full formal
matrix failed the 5% keep gate.
