# 2026-08-25 — direct bucket-gradient infrastructure

## Problem

Gradient views remove unpack copies, but ordinary backward still creates one gradient Tensor per
parameter and then performs 114 pack copies into the final reducer Storage. Both representations
overlap at peak.

## Change

- add a leaf-only `set_grad_accumulation_target` Autograd contract;
- preserve the target's starting values and add every graph contribution in place;
- allow sparse tied-embedding accumulation into an explicitly prepared shared-Storage target;
- reject non-leaf, noncontiguous, mismatched shape/device/dtype targets;
- clear the special contract through ordinary `set_grad` and `zero_grad`;
- after the first reducer step, zero persistent buckets and install their disjoint views before
  backward;
- verify every address/shape/offset after backward and skip all pack copies;
- expose a separate default-off C++/CLI policy and a three-policy Model-S runner.

## Pre-measurement evidence

CPU tests cover preset values, branched and repeated backward, shared Storage embedding, and all
rejections. RCCL tests cover direct view addresses, zero pack/unpack copies, three optimizer steps,
single-global-batch tolerance, and exact rank parameters. The Model-S smoke reaches 114 direct
targets, zero pack/unpack copies, and zero later communication allocations.

The smoke also exposes the counter-hypothesis: leaf accumulation adds work inside backward.
Communication drops from roughly 3.47 to 1.65 ms, while forward/backward rises from roughly 10.22
to 12.95 ms. Only the formal rotated A/B may keep or reject the model route.

Infrastructure gates pass: the three focused CPU Autograd tests, RCCL-labelled `33/33`, and the
machine audit covering 41 graph API entries and 119 native/Python test sources.
