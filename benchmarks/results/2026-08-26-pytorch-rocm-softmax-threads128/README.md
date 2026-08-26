# FP16 typed Softmax, 128-thread candidate

Six fresh MI300X processes run the standard FP16/BF16 ten-case matrix. Only cached
FP16 widths use 128 threads; every other path is unchanged. All correctness, pointer,
ownership and zero-peak-extra gates pass.

At FP16 width4096 the candidate measures 12.027 μs Event and 12.839 μs wall, only
0.424× PyTorch. It is slower than the retained 256-thread baseline and is rejected.
`raw.jsonl` and `summary.json` retain every measurement.
