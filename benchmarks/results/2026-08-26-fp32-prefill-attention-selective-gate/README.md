# Batch-selective prefill Attention model gate

The selective policy keeps B1 on default Attention and uses batch-local operator winners:

- B2: PV 295716;
- B4: QK 311274, PV 295716;
- B8: QK 311303, PV 292462.

It is rejected. All four prefill performance gates pass and global RMS improves 21.6%,
but global Max improves only 6.1% instead of the required 10%. B2 Max/RMS regress.
Peak memory and backend allocation counts are unchanged.

The result contains 16 complete precision processes and 16 paired performance processes.

```bash
python3 benchmarks/single_gpu/fp32_prefill_attention_selective_gate.py \
  --manifest /path/to/hf-models.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory /tmp/selective-attention-gate \
  --runs 2 --performance-warmup 1
```

![selective gate](selective-gate.svg)
