# Version-local BF16 same-algorithm counterfactual

The BF16 plan cache now supports an explicit exact-shape solution registry. Registered plans query
and validate the version-local index, own their required GPU workspace, and pass the algorithm to
hipBLASLt. Unregistered shapes preserve the null-algorithm default.

Solution 75892 makes all 48 B1/B2 stages exact. Fresh no-trace prefill A/B records a 3.77% B1 and
1.27% B2 throughput cost. The registry remains opt-in; no version-local index is hard-coded.

See [Experiment 110](../optimization-log/experiments/110-bf16-same-algorithm.md).
