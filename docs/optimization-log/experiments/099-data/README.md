# Experiment 099 data map

- `profile/r8s4/`: clean continuous-only stdout, Kernel/API/memory stats, raw CSV traces and pftrace.
- `profile/r8s2/`: the matching two-slot profile bundle.
- `paired/r8s4/`: three alternating baseline/scatter-candidate pairs.
- `paired/r8s2/`: three alternating pairs for the higher-step shape.
- `summary.json`: profile attribution and discard decision.
- `gates.json`: profile/schema/full-suite evidence.
- `environment.txt`: device, commands, baseline and pairing contract.

The rejected scatter implementation does not remain in source. Its measurements remain here so the
same idea is not repeatedly proposed without new evidence.
