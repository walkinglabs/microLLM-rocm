# Post-composition T512 phase profile

Experiment 200 profiles the retained explicit composition after grouped QKV
and grouped gate/up are both active.

For each model, rocprofv3 captures load+one and load+six. The committed phase
delta subtracts the first table and divides by five.

| Model | GEMM calls | GEMM time | Total Kernel |
|---|---:|---:|---:|
| Qwen | 217→145, -72 | 3.139→2.657 ms, 1.182× | 5.733→5.680 ms, 1.009× |
| DeepSeek | 253→169, -84 | 6.603→6.007 ms, 1.099× | 10.504→10.160 ms, 1.034× |

The saved calls match the algebra exactly:

- QKV changes three submissions into one, saving two per block;
- gate/up changes two submissions into one, saving one per block;
- 24 Qwen blocks save 72 and 28 DeepSeek blocks save 84.

After composition, GEMM still occupies 46.8%/59.1% of Kernel time. Cast plus
strided materialization occupies another 18.9%/14.8%, and causal softmax is a
visible per-block kernel. The remaining candidate must cross a larger Attention
or cast/layout boundary; another independent projection grouping is not a new
hypothesis.

Files: per-model one/six-step Kernel stats, profile-delta.json, summary.json and
verification.json. The legacy filename three-step-kernel-stats.csv stores the
six-step table.
