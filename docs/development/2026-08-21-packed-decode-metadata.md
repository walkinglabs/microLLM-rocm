# Packed positions-aware decode metadata

When active token IDs originate on CPU, `forward_cached_active_rows()` now packs token IDs,
positions and cache row IDs into one contiguous `[3,A]` Int32 Tensor, performs one H2D copy, and
creates three shared-Storage views on device. Existing device-token callers retain their fallback.

The HIP scheduler gate requires six H2D calls and 76 bytes for the A/B/C scenario. Continuous-only
evidence reduces H2D from 32 to 16 calls at R8/S4 and 56 to 24 at R8/S2 with unchanged 596 bytes.
Alternating Release medians improve 1.033x and 1.065x; all six pairs are positive and checksums,
D2H, D2D and Cache evidence remain unchanged.

See [Experiment 100](../optimization-log/experiments/100-packed-decode-metadata.md) and the
[beginner guide](../dev/packed-decode-metadata.zh-CN.md).
