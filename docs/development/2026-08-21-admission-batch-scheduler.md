# 2026-08-21 — admission batch scheduler

`AdmissionBatchScheduler` groups pending requests by an exact prompt-length/generation/cache-policy
key, invokes public static batch generation per group, and preserves singleton fallback and stable
request order. Requests submitted after a drain wait for the next admission window.

CPU and HIP tests cover multiple groups, singletons, late admission and independent-generation
equivalence. The 1/2/4/8/16 benchmark reaches about 1,260 token/s at HIP B4 and then plateaus because
B4 groups execute serially. This is admission batching, not token-level slot refill.

See [Experiment 074](../optimization-log/experiments/074-admission-batch-scheduler.md).

Final gates: full CPU/HIP 278/278, ASan/UBSan 191/191 and PyTorch-enabled CPU 196/196.
