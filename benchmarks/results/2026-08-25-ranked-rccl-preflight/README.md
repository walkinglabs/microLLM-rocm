# Ranked RCCL resource preflight

Experiment 275 runs from clean revision `9421b6b`.

The non-blocking preflight sees four GPUs for both world2 and world4. `/dev/shm`
has 67,108,864 total and 43,724,800 free bytes. World2 continues through full
training and CPU reference with rank difference zero.

World4 enables official RCCL per-process debug logs. All four ranks fail init and
all four logs contain a shared-memory no-space error. The failed segment is
21,823,872 bytes; RCCL reports version `2.28.3-HEAD:3309c61`.

The failed segment size is not treated as the total required mount size. The
preflight therefore records `required_shared_memory_bytes: null` and
`required_shared_memory_unknown: true`. It reports evidence instead of guessing
a threshold or blocking valid world2 execution.

Four verbose logs total 507,069 bytes. Their structured findings are retained in
`rccl-debug-summary.json`; raw logs are deleted. Current world4 execution remains
unavailable until the container shared-memory resource changes and the full gate
is rerun.
