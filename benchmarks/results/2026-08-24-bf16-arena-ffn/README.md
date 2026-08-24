# Caller-owned BF16 FFN evidence

Experiment 182 executes official Qwen2.5-0.5B and DeepSeek-Distill-1.5B FFN
dimensions with BF16 input/weights/intermediates and an FP32 residual-boundary
output. It compares the existing allocation-returning region, caller-owned Arena
storage, and Arena plus HIP Graph.

The formal matrix contains 54 fresh processes: two models, rows 1/32/512, three
policies, and three process repetitions. Each process excludes three warm-ups and
times twenty regions.

| Shape | Arena eager | Arena Graph | Nodes | Arena bytes |
|---|---:|---:|---:|---:|
| Qwen R1 | 1.063× | 1.182× | 6 | 32,768 |
| Qwen R32 | 1.065× | 1.083× | 6 | 1,048,576 |
| Qwen R512 | 5.548× | 5.049× | 5 | 16,777,216 |
| DeepSeek R1 | 1.038× | 1.068× | 6 | 59,904 |
| DeepSeek R32 | 1.064× | 0.970× | 5 | 1,916,928 |
| DeepSeek R512 | 4.057× | 3.837× | 5 | 30,670,848 |

All complete outputs are bit-exact. Five-node rows support direct BF16×BF16→FP32;
six-node rows use the explicit caller-owned BF16 fallback plus cast. DeepSeek R32
is the required Graph counterexample, so no universal Graph policy is claimed.

The Qwen R512 profile records 130 executed Kernels in every policy. Whole-process
malloc/free changes from 127/126 to 12/11; direct launch APIs change from 129 to
19 under Graph, plus 23 Graph launches.

Files: `raw.jsonl`, `summary.json`, `profile-summary.json`, six rocprofv3 stats
CSVs, and `verification.json`.
