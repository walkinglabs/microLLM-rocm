# 2026-08-19 — repository presentation and developer entry points

## Problem

The framework had strong implementation evidence but presented it as one long status
table. Build prerequisites, compiler versions, repository ownership, operator workflow,
and profiler boundaries were spread across README fragments and chronological records.

## Design influence

The reorganization follows the useful parts of OpenVINO's public repository style:

- a concise product README with direct navigation;
- a developer documentation index;
- a dedicated build entry point;
- an explicit repository/component structure;
- separate add-operation, test, and debug/profile workflows.

The project does not copy OpenVINO branding or adopt its scale. Existing public API and
component names remain stable unless a functional reason requires a change.

## Changes

- replaced the top-level README with a product summary, architecture, quick start,
  measured evidence, weights, performance workflow, repository map, and documentation
  index;
- added `docs/index.md` and `docs/dev/` developer navigation;
- documented minimum CMake/Python/language requirements and exact validated versions of
  GCC/G++, HIP, hipBLASLt, RCCL, rocprofv3, OS, and GPU architecture;
- added `CMakePresets.json` for CPU debug, CPU sanitizers, HIP release, and RCCL release;
- added build troubleshooting, testing/evidence, repository layout, operator development,
  and profiling guides;
- updated contribution instructions to use the same supported entry points.
- moved the root smoke executable into `apps/microllm_runtime_info` and renamed
  course-numbered example targets/tests to framework-purpose names such as
  `microllm_tensor_ppm` and `Examples.CpuHipCompare`.

## Naming decision

`apps`, `examples`, `bindings`, `benchmarks`, `include`, `src`, and `tests` already have
clear industry-standard meanings. They were retained to avoid cosmetic churn. The
course remains isolated on `tutorial/beginner-course`; framework developer docs live in
`docs/dev`; chronological evidence remains in `docs/development`.
