# Contributing

microLLM-rocm accepts small changes with explicit contracts and reproducible
evidence. A directory existing in the repository does not mean its feature is
implemented.

## Before implementation

Create or reference a task contract using [docs/TASK_CONTRACT.md](docs/TASK_CONTRACT.md).
The contract must identify the observed failure, allowed files, public interface,
invariants, reference implementation, tests, and evidence.

## Dependency rules

- `base` depends only on the C++ standard library.
- `core` must remain usable in a CPU-only build.
- low-level operators accept non-owning views and explicit output/workspace/stream;
  they do not own Tensor lifetimes.
- `reference` code never depends on HIP.
- `hip_readable` is retained when a tuned implementation is added.
- optional bindings cannot become dependencies of the core engine.
- distributed code cannot change single-device numerical semantics.

## Local checks

```bash
./scripts/check_cpu.sh
```

HIP changes must additionally run the matching `hip`-labelled CTest tests on a
recorded GPU and ROCm version. Performance changes must include correctness
regression results and raw benchmark metadata.

## Pull requests

Include:

1. the task contract;
2. the unproven assumptions found during review;
3. commands run and their results;
4. one relevant negative or boundary case;
5. status/documentation updates if the evidence state changed.
