# Public CMake package consumer

Date: 2026-08-24
Status: accepted

## Problem

The repository already generated build-tree and relocatable installed CMake Config
packages, exported namespaced component targets, and tested a deliberately broad
internal consumer. That proved the package machinery, but a new user still had to
translate a long README section into their own two-file project. The smallest public
path was documentation text rather than a checked-in, executable example.

## Change

- Added `examples/package-consumer`, an independent CMake project containing only a
  package lookup, one executable, and one imported target link.
- Added `PackageConfig.PublicExample`. It installs the current SDK into a fresh prefix,
  configures the example through `CMAKE_PREFIX_PATH`, compiles it, and runs it.
- Kept the existing build-tree and relocated-install consumers. They still cover all
  exported targets, the C ABI, metadata, missing components, version rejection and
  leakage of repository-only compiler flags.
- Rewrote the README explanation around one idea: the Config file is the installed
  SDK's address card, so consumers do not copy sources or maintain raw include/library
  flags.

## Evidence

| Gate | Result |
|---|---:|
| CPU package tests | 3/3 |
| HIP package tests on gfx942 | 3/3 |
| RCCL package tests | 3/3 |
| CPU Debug full suite | 292/292 |
| ASan/UBSan CPU full suite | 290/290 |
| PyTorch-enabled CPU full suite | 266/266 |
| MI300X/gfx942 HIP-labelled suite | 155/155 |

The public example is intentionally small. It does not prove model quality or GPU
performance; it proves that an unrelated project can discover, compile, link and run
against the produced SDK without repository-local include or library paths.
