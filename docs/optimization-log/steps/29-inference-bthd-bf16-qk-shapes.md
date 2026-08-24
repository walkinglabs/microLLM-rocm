# Step 29 — direct BF16 Q/K sequence and batch matrix

Status: complete; measured shapes retained explicitly

## Evidence

- 60个正式进程覆盖两模型、三case和两策略；
- 六项完整logits位级一致，B2每行top-1一致；
- retained dispatch逐进程精确匹配，控制侧为0；
- peak全部不变；五进程加速范围1.0128×–1.0244×；
- 三进程Qwen B2/T512 1.0091×失败被保留。

## Decision

Close this cast-boundary shape track. Keep explicit/default-off and return to the
causal-softmax hotspot.
