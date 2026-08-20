# Experiment 014 BF16 mixed-GEMM evidence

`shape-benchmark.jsonl` contains alternating FP32 and BF16-mixed rows for five
representative M=1 Linear shapes. BF16 mixed includes activation cast and outputs FP32.

Two shapes improve and three regress, so the next model policy must dispatch per shape.
