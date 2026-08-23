# FP8 native-vs-roundtrip complete-logit runner

Two independent matrix summaries cannot directly compare their logits because the temporary binary
payload is removed after each worker. `hf_fp8_native_roundtrip.py` therefore executes FP32, native
`full`, and `both-roundtrip` inside one fixed experiment boundary and keeps three complete-vector
comparisons per model/context/run.

The runner rotates all three policies through every process-order position, fixes warm-up to zero
and steps to one, and labels all throughput as non-evidence. It verifies the requested diagnostic
mode in worker JSON and rejects any native/fallback FP8 dispatch from `both-roundtrip`.

Its `pairs.jsonl` stores `full_vs_fp32`, `both_roundtrip_vs_fp32`, and
`full_vs_both_roundtrip` over every logit. It also proves that native and software paths used the
same converted-weight and dynamic-activation counts. This is the authoritative Exp142 runner;
separate RMS values are not accepted as a substitute for the direct vector comparison.

Exp142 executed the runner over both official models and T8/T512. Direct native-vs-roundtrip RMS
is 54.8%--76.9% of native total RMS, but native total RMS is never 5% worse than roundtrip total
RMS. Native execution is a material directional perturbation, not a proven total-error driver.
