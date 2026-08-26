# FP16 typed Softmax, retained 1024-thread path

Six fresh MI300X processes run the standard FP16/BF16 ten-case matrix. Only cached
FP16 widths use 1024 threads; every other path is unchanged. All correctness, pointer,
ownership and zero-peak-extra gates pass.

At FP16 width4096 the candidate measures 5.086 μs Event and 5.859 μs wall. It improves
1.076×/1.061× over 512 threads and 1.488×/1.412× over the old 256-thread path, reaching
0.880× PyTorch. It is the retained winner of the 128/256/512/1024 matrix.

`raw.jsonl` and `summary.json` retain every measurement.
