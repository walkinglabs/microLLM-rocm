# Qwen3 BF16 AdamW moment storage

Within the already-rejected BF16-forward path, this compares three steps with FP32 moments against BF16 moments. It measures only the additional optimizer-state compression error.

- moment state: `4,768,399,360 → 2,384,199,680` bytes, exactly half;
- loss added Max: `0.002605`;
- parameter Max/RMS: `3.470e-5 / 1.197e-6`;
- canonical moment Max/RMS: `0.04736 / 3.335e-5`;
- hybrid multi-tensor: 169 tensors / 58,785,792 elements at threshold 1,048,576.

All six fixed gates pass. This is an internal memory/precision trade-off, not PyTorch BF16 alignment. Raw contains 930 final-state records; 14.31GB temporary exports were removed.
