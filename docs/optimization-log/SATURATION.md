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

Each item must start with a new task contract, correctness oracle and track-specific
figure. The FP32 M=1 running best remains frozen until a candidate passes the same fixed
matrix.
