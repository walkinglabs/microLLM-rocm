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
SPLIT_RUNNER = ROOT / "benchmarks/single_gpu/cached_attention_split_matrix.py"
MATERIALIZED_RUNNER = (
    ROOT / "benchmarks/single_gpu/cached_attention_materialized_matrix.py")
FINALIZE_RUNNER = (
    ROOT / "benchmarks/single_gpu/cached_attention_finalize_mapping_matrix.py")
SPLIT_PV_RUNNER = (
    ROOT / "benchmarks/single_gpu/cached_attention_split_pv_matrix.py")


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
finalize_threads = int(a.get("--finalize-threads", "256"))
pv_splits = int(a.get("--pv-splits", "0"))
factor = b * t / 32.0 * (0.97 if dtype == "bf16" else 1.0)
times = {
    "score": 0.020 * factor,
    "softmax": 0.010 * factor,
    "context": 0.030 * factor,
    "pipeline": 0.065 * factor,
    "fused": 0.040 * factor,
    "materialized": 0.030 * factor,
}
if pv_splits:
    split_speedup = {1: 0.75, 2: 1.2, 4: 1.5}.get(pv_splits, 1.0)
    times["split_pv"] = times["materialized"] / split_speedup
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
    "materialized_score_bytes": b * h * t * 4,
    "materialized_max_error": 1.0e-8,
    "materialized_rms_error": 1.0e-8,
    "materialized_bitwise_equal_current": True,
    "materialized_finalize_threads": finalize_threads,
    "materialized_speedup_over_fused": times["fused"] / times["materialized"],
}
if pv_splits:
    record.update({
        "pv_splits": pv_splits,
        "split_pv_probability_bytes": b * h * t * 4,
        "split_pv_partial_bytes": b * h * pv_splits * d * 4,
        "split_pv_max_error": 1.0e-8,
        "split_pv_rms_error": 1.0e-9,
        "split_pv_bitwise_equal_materialized": pv_splits == 1,
        "split_pv_speedup_over_materialized":
            times["materialized"] / times["split_pv"],
    })
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
    allocations = (4 if stage == "split_pv" else 3 if stage == "pipeline"
                   else 2 if stage == "materialized" else 1)
    record[f"{stage}_allocation_calls_per_invocation"] = allocations
    record[f"{stage}_backend_allocation_calls_per_invocation"] = 0
    record[f"{stage}_cache_reuse_calls_per_invocation"] = allocations
print(json.dumps(record))
'''


SPLIT_FAKE = r'''#!/usr/bin/env python3
import json
import sys

a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
b = int(a["--batch"])
t = int(a["--sequence"])
h = int(a["--heads"])
kv = int(a["--kv-heads"])
d = int(a["--width"])
s = int(a["--splits"])
warmup = int(a["--warmup"])
repetitions = int(a["--repetitions"])
speedup = {1: 0.8, 2: 1.1, 4: 1.5}[s]
fused = t / 32.0
split = fused / speedup
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
    "cache_dtype": a["--cache-dtype"],
    "splits": s,
    "order": a["--order"],
    "warmup": warmup,
    "repetitions": repetitions,
    "split_partial_blocks": b * h * s,
    "split_combine_blocks": b * h,
    "split_partial_bytes": b * h * s * (d + 2) * 4,
    "complete_output_accuracy_passed": True,
    "host_to_device_calls": 0,
    "device_to_host_calls": 0,
    "fused_max_error": 1.0e-8,
    "fused_rms_error": 1.0e-9,
    "split_max_error": 2.0e-8,
    "split_rms_error": 2.0e-9,
    "split_speedup_over_fused": speedup,
}
for prefix, value, allocations in (("fused", fused, 1), ("split", split, 3)):
    record[f"{prefix}_event_ms_p50"] = value
    record[f"{prefix}_event_ms_p95"] = value * 1.1
    record[f"{prefix}_wall_ms_p50"] = value * 1.2
    record[f"{prefix}_wall_ms_p95"] = value * 1.3
    record[f"{prefix}_allocation_calls_per_invocation"] = allocations
    record[f"{prefix}_backend_allocation_calls_per_invocation"] = 0
    record[f"{prefix}_cache_reuse_calls_per_invocation"] = allocations
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

        split_fake = root / "fake_split.py"
        split_fake.write_text(SPLIT_FAKE, encoding="utf-8")
        os.chmod(split_fake, 0o755)
        split_output = root / "split-matrix"
        split_completed = subprocess.run([
            sys.executable, str(SPLIT_RUNNER), "--benchmark", str(split_fake),
            "--output-directory", str(split_output),
            "--sequences", "32,64", "--batches", "1",
            "--cache-dtypes", "bf16", "--splits", "1,2,4",
            "--runs", "2", "--warmup", "3", "--repetitions", "4",
        ], text=True, capture_output=True, check=False)
        if split_completed.returncode != 0:
            raise AssertionError(split_completed.stdout + split_completed.stderr)
        split_summary = json.loads(
            (split_output / "summary.json").read_text(encoding="utf-8"))
        split_raw = (split_output / "raw.jsonl").read_text(
            encoding="utf-8").splitlines()
        split_chart = (split_output / "split-search.svg").read_text(
            encoding="utf-8")
        assert split_summary["matrix_complete"] is True
        assert split_summary["process_rows"] == 12
        assert split_summary["candidate_rows"] == 6
        assert split_summary["case_count"] == 2
        assert split_summary["all_case_winners_pass_operator_gate"] is True
        assert {winner["best_splits"] for winner in
                split_summary["winners"]} == {4}
        assert len(split_raw) == 12
        assert "Split-sequence cached Attention search" in split_chart
        assert "best S4" in split_chart

        materialized_output = root / "materialized-matrix"
        materialized_completed = subprocess.run([
            sys.executable, str(MATERIALIZED_RUNNER),
            "--benchmark", str(fake),
            "--output-directory", str(materialized_output),
            "--sequences", "32,64", "--batches", "1,2",
            "--cache-dtypes", "fp32,bf16", "--runs", "2",
            "--warmup", "3", "--repetitions", "4",
        ], text=True, capture_output=True, check=False)
        if materialized_completed.returncode != 0:
            raise AssertionError(
                materialized_completed.stdout + materialized_completed.stderr)
        materialized_summary = json.loads(
            (materialized_output / "summary.json").read_text(encoding="utf-8"))
        materialized_raw = (materialized_output / "raw.jsonl").read_text(
            encoding="utf-8").splitlines()
        materialized_chart = (materialized_output / "comparison.svg").read_text(
            encoding="utf-8")
        assert materialized_summary["matrix_complete"] is True
        assert materialized_summary["process_rows"] == 16
        assert materialized_summary["case_count"] == 8
        assert materialized_summary["all_bitwise_equal_current"] is True
        assert materialized_summary["all_cases_pass_operator_gate"] is True
        assert abs(materialized_summary["minimum_event_speedup"] - 4 / 3) < 1e-12
        assert len(materialized_raw) == 16
        assert "Exact-order materialized-score cached Attention" in \
            materialized_chart

        finalize_output = root / "finalize-matrix"
        finalize_completed = subprocess.run([
            sys.executable, str(FINALIZE_RUNNER), "--benchmark", str(fake),
            "--output-directory", str(finalize_output),
            "--models", "qwen2.5-0.5b,deepseek-r1-distill-qwen-1.5b",
            "--sequences", "32", "--batches", "1",
            "--cache-dtypes", "bf16", "--runs", "2",
            "--warmup", "3", "--repetitions", "4",
        ], text=True, capture_output=True, check=False)
        if finalize_completed.returncode != 0:
            raise AssertionError(
                finalize_completed.stdout + finalize_completed.stderr)
        finalize_summary = json.loads(
            (finalize_output / "summary.json").read_text(encoding="utf-8"))
        finalize_raw = (finalize_output / "raw.jsonl").read_text(
            encoding="utf-8").splitlines()
        finalize_chart = (finalize_output / "mapping.svg").read_text(
            encoding="utf-8")
        assert finalize_summary["matrix_complete"] is True
        assert finalize_summary["process_rows"] == 12
        assert finalize_summary["case_count"] == 2
        assert finalize_summary["all_accuracy_gates_passed"] is True
        assert len(finalize_raw) == 12
        assert {case["winner_threads"] for case in
                finalize_summary["cases"]} <= {64, 128}
        assert "Exact-order finalize mapping search" in finalize_chart

        split_pv_output = root / "split-pv-matrix"
        split_pv_completed = subprocess.run([
            sys.executable, str(SPLIT_PV_RUNNER), "--benchmark", str(fake),
            "--output-directory", str(split_pv_output),
            "--models", "qwen2.5-0.5b,deepseek-r1-distill-qwen-1.5b",
            "--sequences", "32", "--batches", "1",
            "--cache-dtypes", "bf16", "--splits", "1,2,4",
            "--runs", "2", "--warmup", "3", "--repetitions", "4",
        ], text=True, capture_output=True, check=False)
        if split_pv_completed.returncode != 0:
            raise AssertionError(
                split_pv_completed.stdout + split_pv_completed.stderr)
        split_pv_summary = json.loads(
            (split_pv_output / "summary.json").read_text(encoding="utf-8"))
        split_pv_raw = (split_pv_output / "raw.jsonl").read_text(
            encoding="utf-8").splitlines()
        split_pv_chart = (split_pv_output / "split-pv-search.svg").read_text(
            encoding="utf-8")
        assert split_pv_summary["matrix_complete"] is True
        assert split_pv_summary["process_rows"] == 12
        assert split_pv_summary["candidate_rows"] == 6
        assert split_pv_summary["case_count"] == 2
        assert split_pv_summary["all_s1_bitwise_materialized"] is True
        assert split_pv_summary["all_s1_performance_counterexamples"] is True
        assert {case["winner_splits"] for case in
                split_pv_summary["cases"]} == {4}
        assert len(split_pv_raw) == 12
        assert "Exact-softmax split P×V search" in split_pv_chart
    print("cached Attention stage matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
