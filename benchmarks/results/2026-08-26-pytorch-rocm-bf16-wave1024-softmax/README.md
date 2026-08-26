# BF16 1024-thread wave typed Softmax

The retained BF16 cached path changes from a 256-thread shared tree to the same
1024-thread wave reduction already validated for FP16. The predicate remains limited
to cached widths 2048–8192.

Six fresh MI300X processes pass all 10 precision, pointer, ownership and zero-peak-extra
rows. At BF16 width4096, Event/wall changes from 8.701/9.404 μs to 5.157/5.960 μs,
or 1.687×/1.578×. The current caller-owned engine/PyTorch Event ratio is 0.888×.

This does not revive the rejected 256-thread broad-wave candidate. Thread count is part
of the accepted BF16 policy.
