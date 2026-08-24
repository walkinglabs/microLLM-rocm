# Post-BF16-Q/K inference saturation audit

This audit combines the accepted BF16 Q/K profile with the next two rejected
micro-fusions. On the current T512 path, GEMM is 57.4%/66.8% of Qwen/DeepSeek
Kernel time. Perfectly deleting repeat alone could improve Kernel time by only
1.046x/1.035x; perfectly deleting softmax gives 1.111x/1.062x.

The measured 128-thread softmax passes only 4/6 operator rows, and BF16 V
cast+repeat fusion passes only 3/8. The existing readable fused Attention avoids
T-squared storage but runs at about 0.36x because it lacks MFMA tiling and online
data reuse.

The local micro-fusion track is therefore closed. Further inference work needs a
tiled online Attention implementation or a new profile after another subsystem
changes; repeated block-size and one-launch scans are not justified by evidence.
