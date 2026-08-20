# 2026-08-20 — course-only tutorial branch

## Observed problem

`tutorial/beginner-course` was intended to own teaching content, but it still tracked a
snapshot of the complete engine: C++/HIP sources, headers, tests, applications,
bindings, benchmarks, CMake files, result artifacts, and historical development logs.

That made two bad states possible:

1. a learner could unknowingly build an older engine while believing it was current;
2. every framework change would require duplicating hundreds of files into the course.

## Contract

The branch boundary is now:

```text
main                       engine, tests, tools, benchmarks, evidence and dev logs
tutorial/beginner-course   lessons, assignments, small teaching data and course checks
```

The tutorial branch must not track `src/`, `include/`, `tests/`, `apps/`, `examples/`,
`bindings/`, `benchmarks/`, framework scripts, or a top-level CMake project. Lessons use
`MICROLLM_ENGINE_DIR` to reference an independent `main` checkout.

## Change 1 — remove the duplicate engine

Commit `f0fc663` removed 233 framework-mirror files and replaced the old CPU build
workflow with a course structure workflow. It retained 33 course files:

- N0–N8;
- PA0–PA2 and the standalone PA0 arithmetic exercise;
- beginner architecture/operator/task-contract documents;
- small course data and its registry;
- a course-only validator.

The validator checks required lessons, all tracked local Markdown links, and forbidden
engine paths. It intentionally examines tracked files rather than ignored local build
directories.

## Change 2 — update the curriculum

Commit `89f7b3c` added:

- N9: Qwen/DeepSeek config, tokenizer, strict weight loading, complete logits, greedy
  generation, and a real AdamW step;
- N10: FP16/BF16/FP8 storage and compute, E4M3/E5M2 choice, scaled GEMM, accuracy gates,
  and the boundary between operator and whole-model speedup;
- corrected N2/N5/N6 statements after device-native AdamW;
- a current DataParallelTrainer command and its one-process/multi-device limitation;
- explicit links from all runnable lessons to `main` through `MICROLLM_ENGINE_DIR`.

## Verification

After both pushes:

```text
tracked course files       35
Markdown files             22
required files             19
checked local links        25
forbidden engine paths     0
PA0 compile/run            pass
Course structure CI        pass for both commits
remote tutorial head       89f7b3c
```

The course validator does not prove engine correctness. CPU, HIP, HF and RCCL evidence
continues to come only from `main` tests and retained benchmark records.
