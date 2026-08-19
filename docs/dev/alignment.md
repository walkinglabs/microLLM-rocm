# microLLM / PyTorch alignment experiments

The alignment infrastructure runs the same deterministic tiny Transformer in microLLM
and PyTorch, records intermediate values and timings, and generates a machine-readable
plus human-readable comparison.

## One command

Build a CPU runner and use a Python environment containing PyTorch:

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug --target microllm_alignment --parallel

python3 tools/alignment/run.py \
  --microllm-binary build/cpu-debug/apps/microllm_alignment \
  --python /path/to/python-with-pytorch \
  --output /tmp/microllm-alignment \
  --microllm-device cpu \
  --pytorch-device cpu \
  --warmup 5 \
  --repetitions 20 \
  --atol 3e-5 \
  --rtol 3e-5
```

For a microLLM HIP run compared with the available PyTorch reference:

```bash
python3 tools/alignment/run.py \
  --microllm-binary build/hip-release/apps/microllm_alignment \
  --python /path/to/python-with-pytorch \
  --output /tmp/microllm-alignment-hip \
  --microllm-device hip \
  --pytorch-device cpu \
  --atol 2e-3 --rtol 2e-3
```

Use `--pytorch-device cuda` when a working PyTorch ROCm environment is available.
Timing ratios are meaningful only when both sides run on the intended comparable
hardware and mode.

## Three-pass measurement

1. `values`: captures inputs, every autograd forward operator, layer outputs, model
   output, statistics, and full values up to the configured limit.
2. `operator_timing`: disables value copies and records repeated function-level wall
   time. HIP timing synchronizes before and after each range in this correctness-first
   implementation.
3. `layer_timing`: disables operator timers and records embedding, each Transformer
   block, final norm, and full forward time without nested operator instrumentation.

Separating passes prevents value serialization from being counted as operator latency
and prevents nested operator synchronization from inflating the layer timing pass.

## Artifacts

Every output directory contains:

```text
manifest.json
microllm_run.json
pytorch_run.json
microllm_parameters.jsonl
microllm_values.jsonl
pytorch_values.jsonl
microllm_operator_timing.jsonl
pytorch_operator_timing.jsonl
microllm_layer_timing.jsonl
pytorch_layer_timing.jsonl
comparison.json
report.md
logs/{microllm,pytorch,compare}.{stdout,stderr}.txt
```

The manifest records the repository commit/dirty state, command lines, host/tool
versions, devices, seed, warm-up, repetitions, capture limit, tolerances, stage return
codes, and artifact list.

## Numerical report

Records are paired by `kind + name + occurrence`. The comparison checks:

- checkpoint presence;
- dtype and shape;
- truncation state;
- per-element `atol + rtol × |reference|`;
- maximum absolute and relative difference;
- error index and both values at that index;
- mean squared error;
- cosine similarity.

A truncated checkpoint fails by default because sampled values cannot prove full Tensor
alignment. Increase `--max-captured-elements` or explicitly use the comparator's partial
mode only for exploratory diagnosis.

## Timing report

Repeated records are aggregated into minimum, mean, median, p95, and maximum. The report
shows `PyTorch median / microLLM median`; a value greater than one means microLLM was
faster for that measured checkpoint.

Tiny operators are dominated by dispatch and profiler overhead. Do not generalize a
tiny-model speed ratio to Qwen or DeepSeek. Add the target model, shapes, dtype, warm-up,
and end-to-end workload before making a performance claim.

## Extending to a new model

1. Add a microLLM runner that records a complete named parameter trace and stable
   operator/layer checkpoint order.
2. Rebuild the same architecture in the PyTorch runner using those exact weights.
3. Start with one token and one layer; compare after each architectural component.
4. Add prefill, decode, KV cache, and multi-token cases separately.
5. Pin tokenizer/config/weights and record their immutable source revision.
6. Only after value alignment passes, enable timing and optimized candidates.

For Qwen/DeepSeek the current missing architecture work remains listed in
`docs/development/NEXT_STEPS.md`.
