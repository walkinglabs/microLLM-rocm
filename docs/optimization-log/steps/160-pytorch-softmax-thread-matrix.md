# Step 160 — FP16 Softmax thread-count matrix

Status: complete; 1024 kept

FP16 width4096的128/256/512/1024 Event为12.027/7.567/5.472/5.086μs。1024相对512继续提高
1.076× Event、1.061× wall，达到0.880×PyTorch；1024成为当前限定route的默认线程数。

详细记录见[Experiment 344](../experiments/344-pytorch-softmax-thread-matrix.md)。
