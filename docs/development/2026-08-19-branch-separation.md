# 2026-08-19 — framework/course branch separation

## Required ownership

- `main` owns the framework: C++/HIP sources, public APIs, tests, bindings,
  benchmarks, technical contracts, and chronological development records.
- `tutorial/beginner-course` owns N0–N8, PA0–PA2, and the beginner-facing course.

The tutorial branch is based on the tested framework commit so examples can consume the
same engine. Course files are not release gates for the framework branch.

## Main changes

- removed `notebooks/`, `pa/`, and the beginner course document from the `main` tree;
- removed the PA subdirectory from framework CMake;
- changed `scripts/verify_evidence.sh` to validate framework evidence only;
- linked the separate tutorial branch from the framework README and architecture docs;
- retained `docs/development/` on main as required for engineering history.

## Evidence

After separation, framework-only CPU CTest passes 98/98 and the API/test-file coverage
audit still passes. The tutorial branch remains available remotely with all N0–N8 and
PA0–PA2 files.
