# Model-S gradient-ready order audit

Experiment 262 records final leaf-gradient enqueue order without changing the
synchronous reducer. It runs Model-S B1T32 with the selected 25 MiB bucket limit
for three steps in each of three fresh processes and compares both ranks.

- all 9 step records contain the same complete 57-parameter permutation;
- both rank orders match on every step;
- the order is exactly reverse parameter order;
- all three final parameter audits report zero difference.

| Bucket | Parameters | Bytes | Completion | Before backward end? |
|---|---:|---:|---:|---|
| 0 | 0–21 | 26,156,544 | 57/57 (100.0%) | no |
| 1 | 22–55 | 23,605,248 | 35/57 (61.4%) | yes |
| 2 | 56 | 12,582,912 | 1/57 (1.75%) | yes |

Two natural buckets therefore have a structural overlap window. This evidence
admits an Event plus asynchronous all-reduce prototype; it is not itself a speedup
claim and the current training path remains synchronous.
