# Clean continuous profile and scatter rejection

`microllm_bench_scheduler --continuous-only true` now isolates continuous warmup/measurement and
reports scheduler, transfer, allocation and checksum evidence. It explicitly delegates correctness
to the external full suite rather than claiming an in-process comparison it did not run.

Clean R8/S4 and R8/S2 traces show typed GEMM at 61.9%/62.9% of Kernel time, copyBuffer at about
9.3%, positioned decode kernels at 5.84%/7.84%, and exact transfer counters. A proposed batched
logits scatter was correct but alternating A/B medians were 0.993x and 0.973x baseline, so the
operator, Kernel and scheduler route were fully reverted.

See [Experiment 099](../optimization-log/experiments/099-continuous-profile-scatter-discard.md) and
the [beginner profile guide](../dev/continuous-profile.zh-CN.md).
