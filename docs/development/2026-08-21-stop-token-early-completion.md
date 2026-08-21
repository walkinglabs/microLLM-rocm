# Stop-token early completion

Generation now accepts unique vocabulary stop IDs. Single and batched paths append the matching
token and stop before another decode. Batched rows may return different lengths and remain equal to
independent generation; completed rows use ignored dummy inputs only to preserve the current common
cache position.

Schedulers expose `CompletionReason` and count stop completions separately. The B1 reference
scheduler releases KV Cache in the stop step. Static/admission batches still retain a physical row
until the whole group ends, which is now an explicit prerequisite for slot refill rather than an
unstated limitation.

See [Experiment 082](../optimization-log/experiments/082-stop-token-early-completion.md).
