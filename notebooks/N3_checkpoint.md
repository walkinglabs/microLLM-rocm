# N3 — 怎样从同一步恢复训练

## 运行前预测

Suppose AdamW has completed step one. Which fields affect step two?

- model parameters;
- first and second moments;
- optimizer step and hyperparameters;
- global step/data cursor;
- RNG state;
- model and data configuration.

Predict what changes if only parameters are saved.

## 旧办法与失败

A weights-only file can generate text, but it cannot reproduce the next optimizer
update. Reset moments change bias correction and the resumed parameter trajectory.

## 任务契约

```text
Format: fixed magic, explicit version/endian/payload length.
Parameters: unique stable names, shapes, float32 values.
Optimizer: config, step, first moments, second moments.
Experiment: global step, data cursor, serialized RNG, model/data summaries.
Safety: bounded lengths/ranks; verify integrity before mutating live parameters.
Acceptance: resumed next three updates exactly equal uninterrupted updates.
```

## 运行

```bash
ctest --test-dir "$MICROLLM_ENGINE_DIR/build/cpu-debug" \
  --output-on-failure -R CheckpointTest
```

The three tests prove:

1. every state field round-trips;
2. three subsequent AdamW updates are exactly identical after restore;
3. one changed payload byte and duplicate parameter names are rejected.

The dataset cursor has a separate equivalence test:

```bash
ctest --test-dir "$MICROLLM_ENGINE_DIR/build/cpu-debug" \
  --output-on-failure -R TokenDatasetTest.RestoredCursor
```

## 审查点

- A Tensor copy can share Storage; checkpoint snapshots must not alias live moments.
- Parameter order alone is insufficient; names and shapes are checked.
- Loading must validate before replacing parameters.
- A file that can be parsed is not necessarily the correct experiment version.

## 当前边界

The first writer serializes contiguous float32 state and writes directly to the target
path. Atomic temporary-file replacement and mixed-precision states are not yet
implemented. This limitation is reported rather than hidden behind “checkpoint works.”

## 下一步

N4 uses the same Value, optimizer, and checkpoint interfaces to assemble one Decoder-
only Transformer and overfit a generated sequence.
