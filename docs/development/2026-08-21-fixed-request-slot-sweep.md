# Fixed-request 1/2/4/8-slot serving sweep

The `slot-sweep` suite holds eight short requests and eight long requests constant while changing
only `max_slots`. It reports S1-relative speedup, parallel efficiency, allocator peak, exact KV
allocated/active bytes and token differences.

The first 48-process run exposed 18 stable refill failures in S1 and long S2. Fully recycled rows
had position zero but retained reusable backing Storage. `forward_prefill_cached_rows()` incorrectly
selected the first-allocation path whenever all rows were selected. It now selects that path only
when every layer has undefined Storage; recycled full-row admission uses the existing-storage copy
path. Existing CPU and HIP scheduler tests now include single-slot, different-length sequential
refill.

The unchanged matrix then executed 48/48 processes. Qwen is token-exact across all slot counts;
DeepSeek long is exact; DeepSeek short differs between S1/S2 and S4/S8 at request 5, generated token
4. The summary therefore separates `execution_status=pass` from
`status=complete_with_recorded_accuracy_failures`.

See the [beginner guide](../dev/continuous-slot-sweep.zh-CN.md) and
[Experiment 103](../optimization-log/experiments/103-fixed-request-slot-sweep.md).
