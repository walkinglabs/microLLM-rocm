# Step 30 — causal-softmax thread-count counterexample

Status: complete; model policy rejected

## Evidence

- explicit Rows128 primitive passes T256/512/1024 complete outputs;
- T2048 remains on the unchanged 256-thread route;
- 36 operator processes cover Qwen/DeepSeek row shapes;
- only 4/6 Event medians pass 1.01;
- DeepSeek T512 is 1.0071, so the planned model gate is not run.

## Decision

Keep the research primitive, remove model/CLI routing, and close block-size-only
softmax tuning.
