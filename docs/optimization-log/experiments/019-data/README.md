# Experiment 019 evidence

`candidate-run-{1,2,3}.jsonl` contains three independent inference processes for the
narrow-block cached Attention candidate. Each process uses both official models, 2
warm-ups and 5 measured iterations.

The baseline is the retained Experiment 018 state. No profiler was run after the
three-process gate rejected the candidate; spending profiler time cannot rescue an
end-to-end regression. Candidate source was removed before the documentation commit.
