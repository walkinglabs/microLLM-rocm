# 2026-08-25 — Model-S自然多bucket基线

![Model-S data-parallel buckets](../optimization-log/assets/data-parallel-model-s-buckets.svg)

25MiB/3bucket以19.76ms胜出；每卡peak约603.4MB。4MiB/12bucket为28.29ms，1MiB/45bucket
为21.76ms。所有loss和末步参数一致。

这是真实多bucket workload，但还没有overlap。下一步先量化每步pack/unpack copy和临时Tensor，
再决定persistent buffer或gradient-as-bucket view。

