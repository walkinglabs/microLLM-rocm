# FP16 typed Softmax, 512-thread candidate

Six fresh MI300X processes run the standard FP16/BF16 ten-case matrix. Only cached
FP16 widths use 512 threads; every other path is unchanged. All correctness, pointer,
ownership and zero-peak-extra gates pass.

At FP16 width4096 the candidate measures 5.472 μs Event and 6.215 μs wall. It improves
1.383×/1.331× over 256 threads and reaches 0.856× PyTorch, but remains the runner-up
after the 1024-thread measurement. `raw.jsonl` and `summary.json` retain every result.
