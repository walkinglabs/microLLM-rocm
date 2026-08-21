# Experiment 065 evidence

- `baseline-release/`: frozen `main` before BF16 Cache, explicit Release build, 72/72 rows.
- `formal-release/`: same shapes/protocol after the change; only this directory may be used for
  BF16/FP32 speed attribution.
- `short-formal/`: context32, B1/2/4/8, four-token paired PyTorch matrix in the ordinary build;
  useful for continuity with Experiment 064, not the Release long-context attribution.
- `precision/`: complete FP32/BF16 cached-logit comparison; failed rows are expected evidence.
- `profile-before/` and `profile-after/`: aggregate rocprof tables for Qwen T2048 B8.
- `rejected-vectorization/`: exploratory `bfloat162` candidate results retained as a rejection.
- `invalid-build-mismatch/`: first current run used an unspecified build type against a Release
  baseline. It is retained as invalid and cannot support a speed conclusion.

Raw model files and temporary binary logits are not repository artifacts. The checked runner
recreates complete logits and deletes its temporary payload after producing the JSON evidence.
