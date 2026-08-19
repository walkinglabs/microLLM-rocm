# 2026-08-19 — M6 communication/compute overlap experiment

## Contract

Compare identical preallocated work in two schedules:

1. serialized: enqueue compute, wait compute, enqueue all-reduce, wait communication;
2. overlapped: enqueue compute and all-reduce on separate Streams, then wait both.

Use caller-owned output buffers so allocation does not contaminate either measured
schedule. Validate both compute and communication results after timing.

## Failure found

The first multi-GPU compute launch used a GPU0 Stream while the host's current device
remained GPU1, producing `invalid resource handle`. Single-GPU tests could not expose
this. `OpContext::native_stream(Device)` now selects the target HIP device before any
owned or external Stream launch. Low-level TensorView dispatch also selects output
device explicitly.

## Configuration

- two MI300X/gfx942 ranks;
- 1MB all-reduce payload;
- 4,194,304-element preallocated add on each compute Stream;
- three warm-ups and 50 repetitions;
- compute guard 3 and all-reduce sum guard 3.

## Three repeated measurements

| run | serialized ms | overlapped ms | gain |
|---:|---:|---:|---:|
| 1 | 0.16608 | 0.11050 | 33.46% |
| 2 | 0.16444 | 0.11464 | 30.29% |
| 3 | 0.16594 | 0.11593 | 30.14% |

This proves RCCL and independent compute can overlap on the current two-GPU system.
It does not yet prove overlap with the eager backward graph because all gradients
currently become ready after backward completes. The next design step needs per-node
gradient-ready callbacks feeding buckets.

Raw JSONL is committed under `benchmarks/results/2026-08-19-rccl-overlap.jsonl`.
