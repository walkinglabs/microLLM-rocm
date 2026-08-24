# BF16 grouped QKV plus gate/up composition gate

Experiment 197 measures whether the two independently accepted exact policies
compose without registry, Arena, accuracy, setup or memory interference.

Each model runs baseline, QKV-only, gate/up-only and both in three fresh
processes with alternating order. Every process performs two warm-ups and five
measured T512 prefills.

| Model | Baseline | QKV | Gate/up | Both | Both/QKV | Peak ratio |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 93565 | 97741 | 95218 | 99690 tok/s | 1.0199× | 1.00342 |
| DeepSeek | 50328 | 51819 | 50917 | 52711 tok/s | 1.0172× | 1.00173 |

Both versus baseline is 1.0655×/1.0474×. All 24 complete outputs are finite,
inside the BF16 0.25/0.05 envelope and preserve top-1 tokens. Each enabled
policy reports exactly one dispatch per block per forward; disabled policies
report zero.

QKV initialization dominates combined setup at 214.2/205.3 ms. Once QKV has
initialized the grouped library path, gate/up shared-kernel setup is only
0.249/0.239 ms instead of the roughly 57 ms seen in its isolated process.
Combined setup remains an explicit warmed-serving cost, not a one-shot default.

Decision: keep the explicit composed policy. No backend-local indices are
installed by default, and short/batch shapes remain unchanged.

Environment: AMD Instinct MI300X VF, gfx942:sramecc+:xnack-, HIP
runtime/driver 71399004, hipBLASLt 1.3.0. Files: raw.jsonl, summary.json and
verification.json.
