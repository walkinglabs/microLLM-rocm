# Complete-logit and per-block B1/B2 prefill drift

Graph-free inference now emits the same opt-in layer/model TraceSession records as autograd forward.
The ordinary inactive path performs no value copies. `microllm_hf_infer` can write one prefill trace
with an explicit capture limit and refuses ambiguous warm-up/multi-step trace requests.

Three fresh B1/B2 P5 pairs capture all values for embedding, 28 blocks, final norm and complete
last-token logits. Embedding is exact; block 0 is the first nonzero stage. Drift accumulates to
block-27 relative L2 0.006261 and complete-logit max/mean/relative-L2
0.153016/0.028928/0.013777. Duplicate B2 rows are exact at every stage.

See the [beginner guide](../dev/prefill-layer-drift.zh-CN.md) and
[Experiment 106](../optimization-log/experiments/106-prefill-layer-drift.md).
