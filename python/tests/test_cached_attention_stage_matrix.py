#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/cached_attention_stage_matrix.py"


FAKE = r'''#!/usr/bin/env python3
import json
import sys

a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
b = int(a["--batch"])
t = int(a["--sequence"])
h = int(a["--heads"])
kv = int(a["--kv-heads"])
d = int(a["--width"])
warmup = int(a["--warmup"])
repetitions = int(a["--repetitions"])
dtype = a["--cache-dtype"]
order = a["--order"]
factor = b * t / 32.0 * (0.97 if dtype == "bf16" else 1.0)
times = {
    "score": 0.020 * factor,
    "softmax": 0.010 * factor,
    "context": 0.030 * factor,
    "pipeline": 0.065 * factor,
    "fused": 0.040 * factor,
}
record = {
    "schema_version": 1,
    "status": "pass",
    "record_type": "cached_attention_stage_probe",
    "device_name": "fake MI300X",
    "architecture": "gfx942",
    "batch": b,
    "heads": h,
    "kv_heads": kv,
    "sequence": t,
    "width": d,
    "repeats": h // kv,
    "cache_dtype": dtype,
    "order": order,
    "warmup": warmup,
    "repetitions": repetitions,
    "score_elements": b * h * t,
    "context_elements": b * h * d,
    "global_score_bytes": b * h * t * 4,
    "complete_output_accuracy_passed": True,
    "host_to_device_calls": 0,
    "device_to_host_calls": 0,
    "stage_sum_event_ms_p50": sum(times[x] for x in ("score", "softmax", "context")),
    "stage_sum_over_pipeline": sum(times[x] for x in ("score", "softmax", "context")) / times["pipeline"],
    "fused_speedup_over_pipeline": times["pipeline"] / times["fused"],
}
for field in (
    "score_max_error", "score_rms_error",
    "probability_max_error", "probability_rms_error",
    "context_max_error", "context_rms_error",
    "pipeline_max_error", "pipeline_rms_error",
    "fused_max_error", "fused_rms_error",
):
    record[field] = 1.0e-8
for stage, value in times.items():
    record[f"{stage}_event_ms_p50"] = value
    record[f"{stage}_event_ms_p95"] = value * 1.1
    record[f"{stage}_wall_ms_p50"] = value * 1.2
    record[f"{stage}_wall_ms_p95"] = value * 1.3
    record[f"{stage}_allocation_calls_per_invocation"] = 3 if stage == "pipeline" else 1
    record[f"{stage}_backend_allocation_calls_per_invocation"] = 0
    record[f"{stage}_cache_reuse_calls_per_invocation"] = 3 if stage == "pipeline" else 1
print(json.dumps(record))
'''


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake_cached_attention.py"
        fake.write_text(FAKE, encoding="utf-8")
        os.chmod(fake, 0o755)
        output = root / "matrix"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--benchmark", str(fake),
            "--output-directory", str(output),
            "--sequences", "32,64", "--batches", "1,2",
            "--cache-dtypes", "fp32,bf16", "--runs", "2",
            "--warmup", "3", "--repetitions", "4",
        ], text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        raw = [json.loads(line) for line in
               (output / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
        chart = (output / "stage-timing.svg").read_text(encoding="utf-8")
        assert summary["status"] == "pass"
        assert summary["matrix_complete"] is True
        assert summary["case_count"] == 8
        assert summary["process_rows"] == 16
        assert summary["complete_output_accuracy_passed"] is True
        assert summary["zero_payload_transfers"] is True
        assert summary["zero_warm_backend_allocations"] is True
        assert len(summary["cases"]) == 8
        assert len(raw) == 16
        assert {tuple(case["orders"]) for case in summary["cases"]} == {
            ("forward", "reverse")}
        assert all(abs(case["fused_speedup_over_pipeline"] - 1.625) < 1.0e-12
                   for case in summary["cases"])
        assert "Cached Attention: transparent stages vs fused" in chart
        assert "B2 T64 BF16" in chart

        bad = root / "bad.py"
        bad.write_text(
            "#!/usr/bin/env python3\nimport json\nprint(json.dumps({'status':'pass'}))\n",
            encoding="utf-8")
        os.chmod(bad, 0o755)
        rejected = subprocess.run([
            sys.executable, str(RUNNER), "--benchmark", str(bad),
            "--output-directory", str(root / "bad-output"),
            "--sequences", "32", "--batches", "1",
            "--cache-dtypes", "fp32", "--runs", "1",
        ], text=True, capture_output=True, check=False)
        assert rejected.returncode == 2
        assert "schema_version expected" in rejected.stderr
    print("cached Attention stage matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
