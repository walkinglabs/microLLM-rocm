# BF16 grouped gate/up capability gate

Experiment 195 extends the existing grouped-QKV benchmark with a two-operation
gate/up mode. It tests the exact BF16-output shapes used by the retained T512
FFN Arena before any production routing is changed.

Three fresh processes per model screen the first 64 of 10,227 grouped
algorithms. Every screened candidate is finite and bit-exact.

| Model | Stable grouped Event | Device-arguments Event | Reinitialize every call | Setup |
|---|---:|---:|---:|---:|
| Qwen | 1.203× | 1.188× | 0.823× | 0.0537 ms |
| DeepSeek | 1.139× | 1.155× | 0.940× | 0.0524 ms |

Qwen process winners are 65168/65198; DeepSeek selects 65200 in all three
processes. These indices are version-local observations, not a default policy.
The production experiment must select one exact supported index and pass the
complete-model gate.

Reinitializing GroupedGemm for every call is slower on both shapes. The only
accepted next step is a pointer-stable plan using the existing FFN Arena input,
gate, up and persistent block weight addresses.

Decision: capability passes; continue to a separate complete-model integration
experiment. No model or runtime default changes in this node.

Environment: AMD Instinct MI300X VF, gfx942:sramecc+:xnack-, HIP
runtime/driver 71399004, hipBLASLt 1.3.0. Files: raw.jsonl, summary.json, and
verification.json.
