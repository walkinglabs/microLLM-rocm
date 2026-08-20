# Experiment 021 — thread-local HIP device selection cache

Status: `discard`

## Observation

A matched DeepSeek trace contained 30,669 `hipSetDevice` calls. Allocation, deallocation,
Stream and Event lifetime code selected the same `hip:0` repeatedly.

## Hypothesis

Remembering the last successfully selected device per host thread should skip redundant
calls while still issuing `hipSetDevice` whenever a thread switches GPU.

## Candidate and risks

- thread-local integer updated only after successful `hipSetDevice`;
- allocate/free/copy/Stream/Event routes use the helper;
- CPU path unchanged;
- external code calling raw `hipSetDevice` could invalidate the assumption, which makes
  the design less safe for optional framework interop.

Focused allocator, asynchronous copy, Stream, TensorView and model matrix tests passed.

## Profiler result

```text
hipSetDevice calls       30,669 → 1
hipSetDevice API time      2.23 → 0.006 ms
instrumented DeepSeek     29.27 → 31.07 token/s
```

The main hypothesis is real: redundant API calls disappeared.

## End-to-end rejection

The first uninstrumented official matrix moved every workload down:

```text
Qwen generation      154.60 → 141.37 token/s  -8.6%
DeepSeek generation   58.32 →  55.27 token/s  -5.2%
Qwen training        112.43 → 106.90 token/s  -4.9%
DeepSeek training     67.41 →  66.24 token/s  -1.7%
candidate score        1.845199 → 1.750336
```

Both generation rows cross the 5% rejection gate. Exact tokens, losses and observed
parameter updates remain unchanged.

## Decision

`discard`. Lower HIP API counts and a faster instrumented run cannot override the fixed
uninstrumented matrix or the external-device-state risk. Candidate code is removed.

Raw evidence is in [021-data](021-data/README.md).
