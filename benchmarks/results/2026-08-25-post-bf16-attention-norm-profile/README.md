# Post-BF16-Attention-Norm default profile

This repeats the four-process `(six-one)/5` B1T1024 rocprof delta after both
FFN and Attention Norm fusions became BF16 Arena defaults. Every run explicitly
reports both routes enabled.

Qwen/DeepSeek Kernel time is 8.069/14.489 ms, down from 8.208/14.659 ms after
the FFN-only change. Cast calls fall from 72/84 to 48/56, exactly one additional
cast per 24/28 Attention layers. The remaining pair per layer is one FP32->BF16
and one BF16->FP32 boundary; the next step must attribute them before changing
another route.
