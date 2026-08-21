# DeepSeek steady profile and token-history rejection

Experiment 086 profiles the remaining DeepSeek T2048 Release steady-decode gap. Cached Attention
accounts for about 60% of measured decode wall time. B8 additionally shows allocator instability:
large `hipFree` time and exact-size cache reuse collapse.

A caller-owned argmax output and device token history reduced measured D2H calls from 24 to 3 and
kept every token, KV byte and peak-memory contract unchanged. Three alternating Release pairs were
neutral at B1 but produced a 0.861x B8 median. Backend allocations rose from 874 to 13,863 while
cache reuse fell from 15,452 to 2,442. The candidate was fully reverted.

This node changes no retained engine code. It changes the evidence and priority: stabilize B8
allocator/cache behavior first, then retry token-history collection; cached Attention remains the
primary device-kernel optimization track.

See [Experiment 086](../optimization-log/experiments/086-deepseek-steady-profile-d2h-discard.md).
