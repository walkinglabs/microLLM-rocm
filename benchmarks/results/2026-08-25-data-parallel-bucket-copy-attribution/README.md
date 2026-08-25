# Model-S reducer temporary/copy attribution

Experiment 255 records the current clean-commit 25 MiB/3-bucket Model-S path.
Step 1 keeps lazy setup; step 2 is the steady attribution row.

```text
6 rank-local bucket tensors
+ 6 post-all-reduce average tensors
+ 114 unpacked parameter-gradient tensors
= 126 temporary tensors / backend allocations

114 pack copies + 114 unpack copies = 228 D2D copies
15,586,176 parameters × 2 ranks × 3 representations
= 93,517,056 float = 374,068,224 bytes
```

The communication allocation ledger is exactly 126 allocations, 126 backend
allocations, zero cache reuse and 374,068,224 allocated bytes. Non-default RCCL
streams disable the exact-size pool, so these are physical allocations every step.

Steady communication is 7.26 ms of a 22.47 ms step (32.31%). This admits a
persistent reducer-storage design. It must address bucket, average and unpacked
gradient storage together; caching only six bucket tensors would leave 120
allocations and most bytes intact.

