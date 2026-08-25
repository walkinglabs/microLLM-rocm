# Model-S gradient-ready overlap gate

Experiment 263 compares transient, synchronous persistent views, and Event-driven
overlap views in one binary. Every policy uses Model-S B1T32, 25 MiB/3 buckets,
five steps, and a final parameter audit. Three process runs rotate policy order;
steady medians use steps 2–5.

| Policy | Finish/communication | Total | Peak bytes |
|---|---:|---:|---:|
| transient | 6.815 ms | 20.435 ms | 603,383,808 |
| synchronous views | 3.560 ms | 15.025 ms | 636,652,808 |
| overlap views | 1.550 ms | 14.790 ms | 636,652,808 |

Overlap improves total 1.0159x versus synchronous views and 1.3817x versus
transient. All 45 losses and nine final parameter audits are exact; all 12 later
overlap steps enqueue three buckets and allocate no communication Storage.

The candidate remains explicit. It inherits the view plan's 33,269,000-byte peak
cost versus transient, and a single process executes rank0 then rank1 backward.
The next architecture step is one process per GPU, where both ranks can compute
while their ready buckets communicate.
