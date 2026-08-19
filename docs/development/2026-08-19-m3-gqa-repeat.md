# 2026-08-19 — M3 differentiable GQA head repeat

## Contract

Support grouped-query Attention without storing independently projected K/V for
every query head. Repeat each K/V head into its query-head group and reduce repeated
gradients back to the original K/V head.

## Verification

A hand-written `1x2x2` input repeated three times along the head dimension produces
the expected `1x6x2` logical values. Backpropagating a sum returns gradient three to
every original component, proving repeated paths accumulate instead of overwrite.

The operation is generic across positive ranks and dimensions, but its first consumer
is the MHA/GQA Attention module.
