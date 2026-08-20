# 2026-08-19 — framework/course branch separation

## Required ownership

- `main` owns the framework: C++/HIP sources, public APIs, tests, bindings,
  benchmarks, technical contracts, and chronological development records.
- `tutorial/beginner-course` owns N0–N10, PA0–PA2, and the beginner-facing course.

The original tutorial branch was based on a tested framework commit. On 2026-08-20 it
was made course-only: examples now consume a separate `main` checkout through
`MICROLLM_ENGINE_DIR`, preventing a stale engine copy from living in the course branch.
Course files are not release gates for the framework branch.

## Main changes

- removed `notebooks/`, `pa/`, and the beginner course document from the `main` tree;
- removed the PA subdirectory from framework CMake;
- changed `scripts/verify_evidence.sh` to validate framework evidence only;
- linked the separate tutorial branch from the framework README and architecture docs;
- retained `docs/development/` on main as required for engineering history.

## Evidence

After separation, framework-only CPU CTest passed 98/98 and the API/test-file coverage
audit passed. The tutorial branch remains available remotely. The 2026-08-20 follow-up
retains N0–N10 and PA0–PA2 while removing the copied engine tree.
