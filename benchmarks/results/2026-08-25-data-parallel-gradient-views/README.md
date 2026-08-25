# Model-S gradient-as-bucket view gate

Experiment 258 compares transient, persistent-copy, and persistent-view policies
inside one binary. Every policy uses Model-S B1T32, 25 MiB/3 buckets, five steps,
and a final parameter audit. Three process runs rotate policy order; steady medians
use steps 2–5.

| Policy | Unpack Storage/copies | Communication | Total | Live bytes | Peak bytes |
|---|---:|---:|---:|---:|---:|
| transient | 114 / 114 | 6.925 ms | 20.345 ms | 498,757,632 | 603,383,808 |
| persistent-copy | 114 / 114 | 4.045 ms | 15.880 ms | 623,447,040 | 761,342,216 |
| bucket views | 0 / 0 | 3.575 ms | 14.885 ms | 498,757,632 | 636,652,808 |

Views improve communication/total by 1.131x/1.067x versus persistent-copy and
1.937x/1.367x versus transient. All 45 losses and nine final parameter audits
match exactly. Live bytes return to the transient value, and both live and peak
save exactly 124,689,408 bytes versus persistent-copy.

Peak remains 33,269,000 bytes above transient because backward first builds new
per-parameter gradients while the persistent bucket representation is still live.
The policy therefore remains explicit. The next experiment pre-seeds Autograd
with bucket views so backward accumulates directly into the final gradient Storage,
removing 114 pack copies and the overlapping representation.
