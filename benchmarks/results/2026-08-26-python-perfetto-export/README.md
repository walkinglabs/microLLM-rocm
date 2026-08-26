# Python span to Perfetto export

`export_perfetto()` validates Python profile JSONL and atomically writes Chrome Trace
Event JSON. Each span becomes an `X` event with microsecond timestamps, duration,
thread, status, nesting depth, run ID, exception type, and user metadata.

![Perfetto export](perfetto-export.svg)
