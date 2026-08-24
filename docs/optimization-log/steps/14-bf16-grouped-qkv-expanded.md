# Step 14 — expanded grouped-QKV search

Status: complete

## Result

- 64-candidate operator: Qwen 2.010×, DeepSeek 1.692× Event;
- final exact indices: 64713 / 64755;
- steady complete model: 1.0458× / 1.0295×;
- total Kernel phase delta: 1.019× / 1.021×;
- first shared grouped kernel setup: 207.9 / 203.7 ms;
- setup gate: fail; steady policy: keep.

## Boundary

Exact warmed MI300 policy is retained behind explicit registration. Universal and one-shot
defaults remain unchanged. Further work must integrate prewarming with serving admission rather
than silently expanding warm-up.
