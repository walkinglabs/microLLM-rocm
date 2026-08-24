# Direct BF16 Q/K shape pilot

Three fresh processes per policy/model/case cover B1/T256, B1/T1024 and
B2/T512. Correctness, routing and memory all pass. Qwen B2/T512 reaches
`1.0091x`, below the fixed `1.01x` performance gate, while the other five
cases reach `1.0129x` to `1.0233x`.

The pilot is retained as the reason the final matrix uses five processes.
No threshold was changed after seeing this result.
