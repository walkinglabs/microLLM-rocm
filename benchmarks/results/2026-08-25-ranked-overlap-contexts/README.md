# Ranked overlap context-scale matrix

Experiment 271 compares synchronous and overlap bucket views at T32 and T128
from clean revision `31e58e8`. Each context/policy has three fresh two-rank
processes and six steady samples.

| Context | Sync step | Overlap step | Total speedup | Finish speedup | Peak added |
|---:|---:|---:|---:|---:|---:|
| 32 | 8.015 ms | 8.019 ms | 0.9995× | 2.022× | 0 |
| 128 | 9.289 ms | 8.504 ms | 1.0923× | 2.235× | 0 |

T32 reproduces the closed neutral result. T128 passes the 1.01 gate while
keeping current and peak exactly equal between policies. T128 synchronous CV is
6.80% versus overlap 2.21%, so the raw distribution is retained. Removing the
entire slowest process run still gives 9.093/8.504 ms and 1.069×, above the gate.

T128 rank/CPU Max and RMS are 0.003842/2.595e-6 and mean-loss difference is at
most 1.812e-5. T32 retains its prior gates. All ranks match exactly; peer failure
remains bounded; temporary weights and communicator IDs are removed.

Overlap is retained as an explicit context-selective strategy for this measured
Model-S/two-MI300X track: disabled at T32 and admitted at T128. It is not yet a
general default for other models, GPUs, world sizes, bucket limits or contexts.
