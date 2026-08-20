# Experiment 015 raw evidence

These files are retained evidence for a rejected BF16 model policy:

- `extra-shapes.jsonl`: additional M=1 FP32/BF16 mixed-GEMM probes. The
  1×4864×896 and 1×8960×1536 BF16 rows are absent because hipBLASLt returned
  status 6 (unsupported); their FP32 controls remain in the file.
- `bf16-auto-infer-run-{1,2,3}.jsonl`: three independent official-model
  inference processes, each with 2 warm-ups and 5 timed steps.

The candidate cached BF16 copies only for the two shapes that had won in
Experiment 014: 896×896 and 1536×8960. This was an experimental build; the
public precision enum, cache and CLI switch were removed after the speed and
memory gates failed. The JSONL therefore names `bf16-auto`, but released code
does not expose that option.

All runs used the same MI300X VF (`gfx942`) and ROCm runtime/driver version
reported inside each row. Generated token IDs stayed equal to the retained
FP32 path.
