# Experiment 055 evidence

- `pilot.jsonl`: first Qwen saved-probability pair.
- `formal/`: Qwen/DeepSeek × two frameworks × three fresh processes.
- `fallback128.jsonl`: one short-sequence fallback process.
- `profile/`: retained Qwen process aggregates.
- `comparison.json`: speed/memory trade-off and dispatch policy.
- `profile-summary.json`: Experiment 054 before/after row/forward contract.

The first T=256 correctness run failed because the causal upper triangle was not zeroed.
That defect was fixed before all performance records and is retained in the experiment
report as a falsification event.
