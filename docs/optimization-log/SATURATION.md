# Local optimization saturation audit

Date: 2026-08-20 · retained FP32 score: `2.478439`

This does **not** say the framework cannot become faster. It says the current class of
small, local FP32 M=1 decode edits has reached a measured stopping point. Continuing to
permute the same knobs would repeat rejected experiments rather than produce research.

## Search spaces closed by evidence

| Search space | Measured conclusion |
|---|---|
| allocator retirement batch | historical 8/16/24/32 search superseded by immediate default-Stream exact-size reuse in Experiment 087 |
| cached Attention block size | 64/128-thread specialization loses to 256 |
| cached Attention query staging | shared-memory query lowers DeepSeek and score |
| BF16 cached Attention local reads | Key pairs fail official logits; probability normalization and Value pairs are bit-exact but slower |
| local bias fusion | hipBLASLt epilogue and V-bias/store both lose end-to-end |
| Q/K/V grouping | FP32 M=1 GroupedGemm exposes no heuristic |
| hipBLASLt host caching | descriptor and algorithm caches lose repeated matrices |
| explicit GEMM solution | stable micro gain does not survive DeepSeek model gate |
| local gradient reuse | copy-on-write condition never reduces measured allocations |
| device-selection API cache | API count improves, all uninstrumented workloads regress |
| cross-block residual/Norm | 28 fewer launches, Qwen regression lowers score |

## Retained local architecture

- parallel CE and RMSNorm;
- transpose-aware hipBLASLt GEMM;
- device KV cache, direct GQA and fused cached Attention;
- fused Q/K bias+RoPE and paired K/V store;
- two-stage large-vocabulary argmax;
- steady-state exact-size pool with phase-independent legacy-default-Stream reuse;
- width-aware fused residual-Norm.

## Why the next work is architectural

The remaining trace is dominated by repeated GEMM and whole-model scheduling. The next
valid tracks change a larger contract and therefore require new baselines instead of
extending the existing local curve:

1. BF16 activation islands with no permanent FP32+BF16 weight duplicate;
2. packed gate/up or QKV weights loaded directly, not cached as a second model copy;
3. HIP Graph capture with stable addresses and explicit eager fallback;
4. prefill/training Attention forward+backward fusion;
5. autograd liveness planning rather than local `use_count` guesses;
6. separate long-context and batch>1 matrices;
7. multi-GPU overlap measured independently from this single-GPU curve.

Experiment 040 closes one narrower lifecycle question: rebuilding BF16 Linear weights on
every forward is measurably worse than persistent mirrors on both official models. It does
not close the activation-cast or memory problem; the retained option adds 7.9%–10.8% peak
engine memory and therefore cannot be treated as the universal BF16 layout.

Experiment 041 closes the exact eager FFN-island retry: full graph parity and 120 fewer
Qwen allocations produce only a 1.011× same-window throughput ratio. A future island must
change the backward GEMM/storage contract or use another execution model; reintroducing the
same four-parent eager graph node is not a new experiment. It also establishes a measurement
rule: after large shared-GPU drift, an old absolute baseline is invalid until a same-window
control is rerun.

Experiment 043 reopens one previously broad “GEMM already optimized” assumption: the
ordinary forward path was optimized, but wide small-K weight gradients were excluded by the
Auto threshold. That exact space is now closed for Qwen K=3/32/128: K=3 and 32 route to
hipBLASLt when `transpose(left)` produces a wide matrix, while K=128 already did. Further
threshold changes need new exact shapes or registry evidence, not another global cutoff.

Experiment 044 closes the materialized full-sequence Attention graph for the tested
T=3/32/128 Qwen training shapes. Direct GQA, fused causal softmax/context and recomputed
backward are retained. This does not close T=512, head widths above 256, sequence above
4096 or alternative library FMHA backends; those require their own matrix evidence.

Experiment 045 proves the retained training path on DeepSeek 1.5B for T=3/32/128 and batch
2 at T=3. It also closes random initialization and CPU transpose as acceptable official
load behavior. Remaining load work is file decode/streaming/mapping; remaining training
work is the retained optimizer/reduction profile and longer-context matrix.

Experiment 162 closes the Q/K split-half RoPE materialization boundary for attention-bias
models at T512. A layout-aware forward/backward removes 60% of diagnosed strided-copy bytes,
reduces both official-model peaks, and passes throughput non-regression. The remaining 40%
belongs to Value input and context-output layout changes; repeating the Q/K transpose fusion
with different indexing is not a new search space.

Experiment 163 proves that gfx942 hipBLASLt accepts interleaved-head P×V input/output
layouts and that the official T512 operator shapes benefit. It does not close the context
graph boundary: probability-gradient, value-gradient, GQA head reduction and full Autograd
still need matching BTHD contracts before the model can remove those copies.

Experiment 164 supplies those backward contracts and closes the four-layout Attention set
identified by Runtime diagnostics: Qwen/DeepSeek report zero strided-copy calls and zero
Autograd materializations at T512. Further transpose work needs a newly observed layout;
relabeling the now-zero Q/K/Value/context set is not a new optimization hypothesis.

Experiment 166 closes immutable descriptor/layout caching as a production policy for the
interleaved path. Exact operator wall medians improve, but Qwen/DeepSeek T512 model medians
are 0.990×/1.001× against uncached and miss the declared 1.01 gate. The explicit default-off
cache remains diagnostic infrastructure; enabling it by default or changing only the key is
not a new experiment.

Experiment 167 closes moving Attention scale into hipBLASLt alpha as a default policy.
Scale launches disappear and allocations fall, but Qwen throughput regresses and DeepSeek's
fixed parameter changes under the altered FP32 rounding order. Reapplying alpha to Q, scores
or one gradient at a time needs a new numerical hypothesis; merely deleting the same scale
Kernel is not enough.

Experiment 168 closes pairing K/V repeat and reduction Kernels while retaining the same
expanded Tensors. Repeat-family profile time improves, but Qwen falls to 0.976× and
DeepSeek remains below 1.01×. A new GQA candidate must eliminate expansion Storage or use a
different GEMM batch mapping; combining identical traffic in one launch is saturated.

Experiment 169 reopens a narrower GQA space by eliminating expanded Value Storage with
zero batch stride. Universal routing is closed because Qwen and MHA regress; width-128
DeepSeek P×V improves 1.60×. Only a width-selective full-backward experiment remains open.

Experiment 170 rejects that complete P×V+dP route: DeepSeek allocation falls but extra
KV-group dP GEMMs keep dispatch count flat and raise Kernel time. Only forward-only P×V
remains distinct; it retains the old one-call dP backward.

Experiment 171 rejects that final forward-only route as well. DeepSeek remains flat,
parameter equality changes, and profile dispatches stay constant because each removed copy
becomes an extra GEMM. The current zero-stride model-routing family is fully closed.

Experiment 172 reopens local gradient reuse only after source-aware diagnostics identify
72/84 genuinely exclusive Qwen/DeepSeek destinations. The implementation saves exactly
144/168 engine allocations over two steps, but the allocator cache means backend allocation,
HIP allocation/free, add Kernel count and peak are unchanged. Throughput reaches only
1.0042×/0.9952× and misses the declared two-model 1.01 gate. Another `use_count` predicate is
not a new hypothesis; future work must remove add computation or plan lifetimes graph-wide.

Experiment 173 confirms that the architectural submission track is materially different from
the saturated local edits. Caller-owned HIP Graph replay improves every 32–512-node row by
1.207×–1.909× while preserving device Kernel count; one/eight-node counterexamples prevent a
universal policy. The runtime primitive is retained, but this does not reopen local M=1 decode:
model integration now requires explicit Stream propagation and stable graph-wide Storage.

Experiment 174 narrows that open Graph track. A caller-owned hipBLASLt output is capture-safe and
bit-exact, yet Qwen improves only at 32 repeats and DeepSeek remains below 1.0 even there. Host
submission falls while 322 profiled Kernels remain 322. Repeating a vendor GEMM is closed; only a
heterogeneous captured region with planned addresses remains a distinct model hypothesis.

Experiment 175 rejects Stream routing without lifetime routing. A scoped override passes local
caller-owned tests but changes complete tiny-model logits by Max 1.412–3.846 and leaves the next
embedding launch with a prior-capture error. The API is removed. The Graph track is blocked on a
new ownership mechanism—deferred release or an activation arena—not another context wrapper.

Experiment 176 supplies that ownership mechanism at the explicit-Stream level. Exact temporary
chains improve 2.28×–2.74× versus synchronizing before every safe free; Stream synchronizations
fall 320→10 with unchanged work. It retains up to 2.08 MB in the measured matrix, so model reuse
must pass both correctness and pending-memory gates. The primitive reopens one model-Stream retry,
not arbitrary Graph capture.

Experiment 177 completes that one permitted model-Stream retry. Binding otherwise-default
operators, runtime strided copies and deferred release restores bit-exact Qwen/DeepSeek inference,
loss and parameter updates in 48/48 processes. Performance fails every row: inference is
0.125×–0.862× and training 0.235×–0.575×, with up to 15.6 GB deferred. Profiler Kernel count is
identical while allocation/free duration rises sharply because non-default Streams disable the
legacy-default-Stream-only exact-size pool. Lexical routing and larger deferred tables are now
closed; only a same-Stream ordered allocator or planned activation arena is a distinct retry.

Experiment 178 closes a separate multi-GPU ownership defect discovered by the expanded gate.
Static hipBLASLt handles cannot cross device indices. Per-device handles restore the existing
RCCL set from 6/11 to 11/11, while four Qwen/DeepSeek T512 single-GPU ratios remain
0.998×–1.023×. Reintroducing one process-wide handle or fixing the symptom with global
synchronization is closed; future multi-GPU work may assume vendor handles are rank-device local.

Experiment 179 tests the ordered-allocation handoff from Experiment 177. Eager async allocation
reuses exactly two addresses but reaches only 0.619×–0.709× and reserves 128 MiB high-water.
Captured allocation/free nodes reach only 0.036×–0.048×, own N addresses and create `3N+1` nodes.
Both policies are closed; the explicit Beta primitive remains for conformance. Only a stable
caller-owned activation arena plus graph-wide liveness is still distinct on this track.

Experiment 180 validates that final micro-level distinction. A two-slot stable arena improves
eager chains 1.071×–1.768× and compute-only Graph replay 1.314×–3.066× with exact outputs.
Graph setup is 14–16 ms, so break-even spans 9–1,280 replays. Allocator selection is now closed;
only mapping a real heterogeneous model region into stable arena offsets remains open.

Experiment 181 maps the official dense FFN dimensions into that arena. Qwen R32/R512 and
DeepSeek R512 Graph rows improve 1.202×/2.970×/1.679×; DeepSeek R32 reaches only 1.005× and closes
universal routing. The FP32 region is retained. Remaining work is narrower: caller-owned BF16
output contracts and a complete model-logit gate, not another FP32 arena shape sweep.

Experiment 182 closes that BF16 primitive gap without deleting unsupported shapes. A caller-owned
fallback makes Qwen R1/R32 and DeepSeek R1 six-node Graphs; direct-output rows remain five nodes.
All 54 outputs are exact, eager Arena passes five rows, and Graph passes five rows. DeepSeek R32
Graph is 0.970×, so universal Graph routing is closed. Only eager Arena complete-model routing is
open; another operator-only BF16 shape sweep is not a distinct hypothesis.

Experiment 183 carries the eager Arena through complete Qwen/DeepSeek logits, cache decode and
five context/batch cases. All 60 outputs are exact and allocation falls, but only three of ten
rows pass 1.01. Universal model routing is closed. Both T512 rows improve 1.020×–1.022×, so only
the model-independent `flattened rows>=512` predicate remains distinct. Per-model decode tuning,
universal routing and model Graph are not supported by this evidence.

Experiment 184 validates the only remaining FFN selection hypothesis. With `minimum_rows=512`,
both eligible official rows improve 1.019×–1.022×; eight shorter rows have zero Arena entries and
exact baseline allocation/peak counters. The threshold is retained and further FFN crossover
search between unmeasured points is closed. The next liveness work must target another region,
such as shared-cast BF16 Q/K/V, with the same full-model gate.

Experiment 185 applies that gate to shared-cast Q/K/V on top of the retained FFN policy. Allocation
falls again, yet eligible T512 ratios are only 1.004×/1.005× and model routing is rejected. QKV
persistent-storage optimization is locally saturated. A new liveness candidate now requires
allocation-size/source attribution; moving another manually guessed Tensor family is closed.

Experiment 186 supplies that attribution. Qwen/DeepSeek T512 independently select
`attention.core` as the largest source at 572.5/792.7 MB and 53.0%/43.6% of logical bytes. The
diagnostic distribution is identical across three processes per model. Allocation-source search is
closed; only exact Attention core liveness/out work is justified next.

Experiment 187 performs that exact liveness work. The largest source drops by 600/700 model
allocation calls, yet T512 improves only 1.004×/1.002× and retained backing raises peak. Attention
core Arena joins QKV as a rejected model policy. The persistent-Storage/liveness track is saturated;
future Attention work must improve device computation or algorithms.

Experiment 188 opens that device-math track with evidence: exact FP32 QK/PV hipBLASLt solutions
improve all four official T512 operator shapes by 1.114×–1.324× after complete-output checks.
Inventory is complete; only exact registry plus model gating remains open. Generic algorithm
hard-coding or extrapolation to other versions/shapes is closed.

Experiment 189 closes that exact-solution track. The fastest approximate QK indices pass isolated
checks but accumulate complete-logit Max 0.07290/0.04437, so the formal gate switches to the
fastest bit-exact alternatives. All 24 final processes are bit-exact and peak/allocation neutral.
QK/PV/both reaches 1.009/1.004/1.008 on Qwen and 0.999/1.003/1.004 on DeepSeek; no policy passes
the two-model 1.01 gate. Exact registry infrastructure remains, all defaults remain unchanged, and
another index sweep is closed. A future Attention compute retry must fuse a larger surrounding
region or present a new profile-selected mechanism.

Experiment 190 uses a new phase-delta trace rather than the misleading whole-process profile.
Incremental GEMMs are 53.6%/61.9% of Qwen/DeepSeek T512 Kernel time. BF16 grouped QKV is available
only with BF16 outputs; pointer-stable cached plans improve operator Event 1.881×/1.225×, while
per-call initialization is 0.908×/0.815×. Complete models stay within the declared BF16 envelope,
but Qwen/DeepSeek reach 1.032×/1.001× and peak rises 0.34%/0.17%. The explicit primitive remains;
the two-model default and any model-name dispatch are closed. A shape policy needs additional
checkpoints before it becomes a distinct hypothesis.

Experiment 191 reopens only the candidate-count assumption and finds stable exact indices
64713/64755. Operator Event becomes 2.010×/1.692× and steady complete-model throughput becomes
1.046×/1.030×. Device user arguments reduce 24/28 grouped initializations to one shared kernel;
phase delta confirms 48/56 fewer GEMM calls and total Kernel gains of 1.019×/1.021×. The remaining
kernel setup is 207.9/203.7 ms, above the 100 ms default gate. The warmed serving policy is kept;
one-shot default is closed until scheduler-level pre-admission warmup is measured.

Experiment 192 supplies the pre-admission lifecycle without changing the default. Zero-warmup
fresh processes show ordinary BF16 first forward already costs about five seconds. Lazy grouped
first request is 5744/5741 ms; explicit prewarm costs 915/886 ms and the admitted request becomes
4852/4795 ms. Combined cost stays near lazy total. Prewarm API is retained and repeated rows are
no-ops. Startup optimization remains open; moving setup is not removal.

Experiment 193 closes broad hipBLASLt all-kernel preload as that startup optimization. FP32 first
use already costs about 3.6 seconds, while BF16 lazy costs about 5.0 seconds. Preload-all raises
Qwen/DeepSeek first forward to about 17.2 seconds, a 3.417×/3.447× slowdown, and process wall
slows 3.140×/2.938× without changing engine peak. A future startup candidate must select exact
used kernels or change process/lifecycle ownership; broad preload and another wrapper around full
forward warm-up are closed.

Experiment 194 tests that exact-selection exception and closes the one-shape solution shortcut.
Three tuner processes per model select common-passing BF16-output gate/up indices with
1.059×/1.032× operator Event gains. In 24 fresh model processes, cold ratios are
0.990×/0.996×, process-wall ratios are 0.978×/0.981× and steady ratios are 0.973×/1.007×.
Logits are bit-exact and peak is unchanged, so this is a clean performance rejection. Further
startup work needs library/module lifecycle control or process persistence, not another first-GEMM
solution index.

Experiment 195 opens a distinct steady FFN submission track. Two-operation BF16 GroupedGemm has
10,227 available algorithms for both official T512 gate/up shapes; 64/64 screened candidates are
bit-exact. Device-user-arguments Event ratios are 1.188×/1.155×, while per-call reinitialization
is only 0.823×/0.940×. The capability benchmark is retained. Only pointer-stable FFN Arena
integration is open; a stateless grouped function or model-name dispatch is not supported.

Experiment 196 completes that integration. One initialized kernel plus 24/28 device-argument plans
improves uninstrumented T512 throughput 1.0176×/1.0117× with top-1 preserved, BF16 Max/RMS inside
0.25/0.05 and peak ratios 1.000008×/1.000003×. Phase delta removes exactly one GEMM submission
per block and improves GEMM time 1.035×/1.020×; instrumented DeepSeek total Kernel remains a
0.998× counterexample. The explicit exact policy is retained. Version-local defaults, non-Arena
routing and unmeasured short/batch shapes remain closed.

Experiment 197 proves the two exact grouped families compose rather than shadow each other.
Both/base reaches 1.0655×/1.0474× and both/QKV adds 1.0199×/1.0172× in 24 fresh processes;
each enabled registry reports its exact dispatch count and disabled sides report zero. Peak ratios
are 1.00342×/1.00173× and complete BF16/top-1 gates pass. QKV initialization amortizes gate/up
kernel setup to below 0.25 ms, but combined setup remains 214.5/205.6 ms. Explicit T512
composition is retained; broader shape policy and one-shot defaults remain open/disabled.

Experiment 198 supplies operator evidence for flattened rows 256/1024. All eight model/rows/
projection cases have 10,227 algorithms, 64/64 passing candidates and device-arguments Event
ratios from 1.124× to 1.695×. A one-process rows256 reinitialization apparent win becomes 0.964×
after three processes, preserving the stable-address design. Operator capability is kept; only
B1/T256, B1/T1024 and B2/T512 complete-model routing is open.

Experiment 199 closes that complete-model gate. All six model/workload cases pass at
1.0212×–1.1075× with per-batch top-1, BF16, setup and peak gates. B1/T1024 and B2/T512 share
rows1024 keys but show different ratios, confirming workload identity cannot collapse to flattened
rows. The first run exposed and stopped on a CLI B2 export bug; last/full modes now write every
batch row and a real tiny-HF binary fixture guards it. Explicit rows256/1024 policies are retained.

Experiment 200 profiles the final composition. Qwen/DeepSeek GEMM calls fall 217→145 and 253→169,
exactly 72/84 saved submissions; GEMM time improves 1.182×/1.099× and total Kernel
1.009×/1.034×. GEMM remains 46.8%/59.1%, with cast plus strided materialization another
18.9%/14.8%. Independent-projection grouping is locally saturated. The next candidate must cross
a larger Attention or cast/layout boundary, not add another stateless grouped or exact-index edit.

Each item must start with a new task contract, correctness oracle and track-specific
figure. The FP32 M=1 running best remains frozen until a candidate passes the same fixed
matrix.
