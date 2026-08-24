# Step 23 — post-composition profile

Status: complete

## Evidence

- four rocprofv3 processes and ten derived steady forwards;
- GEMM calls 217→145 and 253→169;
- GEMM time speedups 1.182×/1.099×;
- total Kernel speedups 1.009×/1.034×;
- remaining GEMM share 46.8%/59.1%;
- remaining cast plus strided share 18.9%/14.8%.

## Decision

Grouped independent-projection submission is locally saturated. Select a larger Attention or
cast/layout boundary next.
