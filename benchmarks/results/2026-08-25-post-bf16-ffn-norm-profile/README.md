# Post-BF16-FFN-Norm default profile

This repeats the load-subtracted B1T1024 profile after BF16 FFN Arena made
direct Norm output the default. For each model, rocprof records load+1 and
load+6 prefills; `(six-one)/5` removes load and plan setup.

Qwen/DeepSeek Kernel time is 8.208/14.659 ms versus 8.315/14.862 ms before the
change. FP32/BF16 casts fall from 96/112 to 72/84 per step, exactly one removed
cast per 24/28 FFN layers. GEMM now occupies 60.9%/68.2%. The next bounded
question is the remaining Attention projection-input boundary, not another FFN
activation tweak.
