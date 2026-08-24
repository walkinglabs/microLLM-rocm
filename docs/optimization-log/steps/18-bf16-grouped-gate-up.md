# Step 18 — grouped gate/up capability

Status: complete

## Evidence

- six fresh processes across official Qwen/DeepSeek T512 shapes;
- 10,227 available algorithms and 64/64 bit-exact screened candidates;
- stable grouped Event ratios 1.203×/1.139×;
- device-user-arguments Event ratios 1.188×/1.155×;
- per-call reinitialization ratios 0.823×/0.940×;
- argument setup medians below 0.054 ms.

## Decision

Keep the benchmark capability and proceed to pointer-stable FFN Arena integration. Production
routing remains unchanged until complete-model gates pass.
