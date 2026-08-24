# Step 07 — in-place gradient accumulation and buffer reuse

Status: `complete` — local reuse variants measured; graph-wide planning is a new track

## Hypothesis

Allocating a new Tensor for every gradient branch accumulation contributes to training
allocation churn and launch count.

## Design

- add validated `add_` for same-shape/device/dtype Tensor;
- allocate leaf grad buffers lazily once;
- distinguish set-to-none from zero-in-place;
- release intermediate gradients after their backward closure;
- optionally assign reusable arena slots using liveness;
- preserve repeated-backward semantics.

## Required tests

- shared branch accumulation;
- same parameter used many times;
- repeated backward;
- view/transpose gradient order;
- zero_grad modes;
- full named Transformer gradients;
- no alias between parameter data and gradient.

## Falsification

If allocations fall but training time does not, serial CE/Norm or transpose copies still
dominate. Do not add a complex planner before simple in-place accumulation is measured.

## Keep gate

Correctness unchanged, allocation/launch count reduced, training rows improve or memory
drops materially without unexplained throughput regression.

## Experiment 010 result

A local `Storage::use_count()==1` condition was correct but reduced zero measured
allocations: Qwen/DeepSeek remained 9,200/10,715. Backward gradients commonly still
alias upstream nodes. The candidate was removed. Future work must explicitly model
contribution count/liveness; this step is not complete.

## Experiment 046 handoff

The retained DeepSeek `1×128` process trace changes the priority. AdamW launches 1,017
times, exactly 339 parameters × three traced steps, and occupies 32.94% of Kernel time.
A persistent multi-tensor pointer table cannot safely reference gradients while
`zero_grad()` destroys their Tensor allocations. The next candidate therefore keeps the
buffer address stable while separately tracking whether a current gradient is valid.

This is narrower than a general liveness planner and directly enables the measured next
optimization. It must prove repeated-backward semantics, address stability, no parameter
alias, and zero optimizer payload transfers before multi-tensor AdamW is attempted.

## Experiment 047 result

The stable-address candidate passed those correctness tests but failed the matched Qwen
`1×128` throughput gate: `802.70→757.48 token/s` (`−5.63%`). Peak memory fell 5.21%, but
copying every first leaf contribution added a memory pass and launch. The implementation
was removed.

The revised optimizer design does not require stable addresses. It passes the current
parameter/gradient/moment pointers in bounded groups as Kernel arguments. Sixteen tensors
per launch keeps metadata bounded and should reduce 339 launches to about 22 without a
pointer upload or gradient copy. This alternative must still pass scalar AdamW equality
and end-to-end keep gates.

## Experiment 048 result

Passing fresh addresses in 16-Tensor Kernel arguments passed scalar CPU/HIP state equality.
Grouping all 290 Qwen tensors reduced launches to 19 but regressed `1×128` throughput by
42.3%, so the matrix early-stopped. Grouping only 121 tensors with at most 4,096 elements
reduced launches to 177; four-shape speedups were `0.988×–1.027×`, with unchanged memory.
No row met the 5% gate, and the code was removed.

This falsifies launch count as the primary AdamW explanation. The retained profile time is
mostly bytes moving through large parameter/gradient/moment tensors. The next candidate
must benchmark vector-width/coalescing on exact large shapes before model integration.

## Experiment 049 result and saturation boundary

The exact-shape benchmark covers scalar/float4, width 8, sqrt/rsqrt, and mirror/no-mirror.
Float4 wins 5.6%–19.4% on three mirrored counts, but the corresponding no-mirror output
weights regress and an explicit all-parameter Qwen pilot regresses every formal shape.
Width 8 and corrected rsqrt also lose.

The float4 implementation remains explicit for research and benchmarking; `Auto` remains
Scalar. Step 07 is complete at this boundary: three gradient-buffer/multi-tensor designs
and the local vector-width space have been measured. Reopening it requires new trace
evidence or a different optimizer-state representation, not another unmeasured launch
rewrite.

## Experiment 172 source-aware retry and final boundary

New diagnostics finally found a narrower case that Experiment 010 could not see: Qwen and
DeepSeek have 72/84 exclusive contiguous reshape-gradient destinations per T512 step. A safe
in-place primitive removes exactly 144/168 engine allocation calls over two steps and passes
shared-graph/overlap/device correctness.

The model gate still rejects it at `1.0042×/0.9952×`. Qwen rocprofv3 shows unchanged backend
allocation, HIP allocation/free, add Kernel, total Kernel and peak counts because the existing
exact-size cache already handles those temporary blocks. Local owner predicates are now closed.
Reopening this step requires graph-wide lifetime/buffer planning or eliminating the add work,
not another `Storage::use_count()` condition.
