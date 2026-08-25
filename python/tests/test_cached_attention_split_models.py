#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/compare_cached_attention_split_models.py"
MATRIX_RUNNER = ROOT / "benchmarks/single_gpu/materialized_attention_model_matrix.py"


FAKE = r'''#!/usr/bin/env python3
import array
import json
import sys

a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
b = int(a["--batch"])
n = int(a["--new-tokens"])
steps = int(a["--steps"])
warmup = int(a["--warmup"])
splits = int(a["--cached-attention-splits"])
materialized_explicit = "--cached-attention-materialized" in a
materialized = a.get("--cached-attention-materialized", "true") == "true"
candidate = bool(splits) or materialized
minimum = int(a["--cached-attention-minimum-sequence"])
tokens = [int(value) for value in a["--tokens"].split(",")]
values = [float(index) / 32.0 + (1.0e-6 if splits else 0.0)
          for index in range(b * 16)]
with open(a["--cache-logits-output"], "wb") as stream:
    array.array("f", values).tofile(stream)
record = {
    "parameter_count": 1234,
    "token_count": len(tokens),
    "batch": b,
    "decode_tokens": n,
    "warmup": warmup,
    "steps": steps,
    "use_cache": True,
    "cache_prefill_mode": "full",
    "decode_mode": "steady",
    "decode_step_semantics": "one_model_forward_per_measured_token",
    "kv_cache_dtype": a["--kv-cache-dtype"],
    "requested_cache_capacity": int(a["--cache-capacity"]),
    "kv_cache_capacity_tokens": int(a["--cache-capacity"]),
    "kv_cache_active_tokens": len(tokens) + n,
    "kv_cache_actual_bytes": 4096,
    "kv_cache_active_bytes": 4096,
    "cached_attention_splits": splits,
    "cached_attention_minimum_sequence": minimum,
    "cached_attention_materialized_scores": materialized,
    "cached_attention_materialized_policy": (
        "explicit-on" if materialized_explicit and materialized
        else "explicit-off" if materialized_explicit else "auto-enabled"),
    "cached_attention_materialized_auto_eligible": not materialized_explicit,
    "measured_tokens": b * n * steps,
    "measured_forward_steps": b * n * steps,
    "generated_tokens": list(range(n)),
    "decode_tokens_per_second": 150.0 if candidate else 100.0,
    "engine_peak_bytes": 1020 if candidate else 1000,
    "engine_allocation_calls": 130 if candidate else 100,
    "engine_backend_allocation_calls": 10,
}
print(json.dumps(record))
'''


FAKE_COMPARISON = r'''#!/usr/bin/env python3
import json
import pathlib
import sys

a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
context = int(a["--context"])
batch = int(a["--batch"])
speedup = 1.02 if context == 512 else 1.20
output = pathlib.Path(a["--output-directory"])
output.mkdir(parents=True, exist_ok=True)
summary = {
    "status": "pass",
    "model": a["--model"],
    "revision": "fixture-r1",
    "context": context,
    "batch": batch,
    "candidate_policy": "materialized",
    "accuracy_gate_passed": True,
    "performance_gate_passed": context >= 2048,
    "all_generated_tokens_equal": True,
    "maximum_logit_error": 0.0,
    "maximum_logit_rms_error": 0.0,
    "median_current_throughput_tokens_per_second": 100.0,
    "median_split_throughput_tokens_per_second": 100.0 * speedup,
    "median_throughput_speedup": speedup,
    "paired_speedups": [speedup, speedup, speedup],
    "leave_one_pair_out_speedups": [speedup, speedup, speedup],
    "median_peak_bytes_delta": 0,
    "median_allocation_calls_delta": 10,
    "median_backend_allocation_calls_delta": 1,
}
(output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
print(json.dumps(summary))
'''


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config = root / "config.json"
        config.write_text(json.dumps({"vocab_size": 16}), encoding="utf-8")
        weights = root / "weights.bin"
        weights.write_bytes(b"fixture")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "models": [{
                "name": "fixture",
                "revision": "fixture-r1",
                "parameter_count": 1234,
                "config": str(config),
                "weights": str(weights),
                "inference": {"token_ids": [1, 2, 3]},
            }],
        }), encoding="utf-8")
        fake = root / "fake_model.py"
        fake.write_text(FAKE, encoding="utf-8")
        os.chmod(fake, 0o755)
        output = root / "output"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--manifest", str(manifest),
            "--binary", str(fake), "--output-directory", str(output),
            "--model", "fixture", "--context", "8", "--batch", "2",
            "--decode-tokens", "4", "--cache-dtype", "bf16",
            "--splits", "4", "--minimum-sequence", "4",
            "--warmup", "1", "--steps", "2", "--runs", "3",
        ], text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        raw = (output / "raw.jsonl").read_text(encoding="utf-8").splitlines()
        pairs = (output / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        chart = (output / "comparison.svg").read_text(encoding="utf-8")
        assert summary["status"] == "pass"
        assert summary["candidate_policy"] == "split"
        assert summary["process_rows"] == 6
        assert summary["pair_rows"] == 3
        assert summary["median_throughput_speedup"] == 1.5
        assert summary["all_generated_tokens_equal"] is True
        assert summary["accuracy_gate_passed"] is True
        assert summary["performance_gate_passed"] is True
        assert summary["median_peak_bytes_delta"] == 20
        assert summary["median_allocation_calls_delta"] == 30
        assert summary["median_backend_allocation_calls_delta"] == 0
        assert summary["maximum_logit_error"] < 1.1e-6
        assert len(raw) == 6
        assert len(pairs) == 3
        assert "Official model · current vs split cached Attention" in chart
        assert "median speedup  1.5000x" in chart

        auto_output = root / "auto-output"
        auto_completed = subprocess.run([
            sys.executable, str(RUNNER), "--manifest", str(manifest),
            "--binary", str(fake), "--output-directory", str(auto_output),
            "--model", "fixture", "--candidate-policy", "auto",
            "--context", "8", "--batch", "2", "--decode-tokens", "4",
            "--cache-dtype", "bf16", "--minimum-sequence", "4",
            "--warmup", "1", "--steps", "2", "--runs", "3",
        ], text=True, capture_output=True, check=False)
        if auto_completed.returncode != 0:
            raise AssertionError(auto_completed.stdout + auto_completed.stderr)
        auto_summary = json.loads(
            (auto_output / "summary.json").read_text(encoding="utf-8"))
        assert auto_summary["candidate_policy"] == "auto"
        assert auto_summary["accuracy_gate_passed"] is True
        assert auto_summary["performance_gate_passed"] is True

        fake_comparison = root / "fake_comparison.py"
        fake_comparison.write_text(FAKE_COMPARISON, encoding="utf-8")
        os.chmod(fake_comparison, 0o755)
        matrix_output = root / "matrix"
        matrix_completed = subprocess.run([
            sys.executable, str(MATRIX_RUNNER),
            "--comparison-runner", str(fake_comparison),
            "--manifest", str(manifest), "--binary", str(fake),
            "--output-directory", str(matrix_output),
            "--models", "fixture,fixture2", "--contexts", "512,2048",
            "--batches", "1,2", "--decode-tokens", "4",
            "--runs", "3", "--warmup", "1", "--steps", "2",
        ], text=True, capture_output=True, check=False)
        if matrix_completed.returncode != 0:
            raise AssertionError(
                matrix_completed.stdout + matrix_completed.stderr)
        matrix = json.loads(
            (matrix_output / "summary.json").read_text(encoding="utf-8"))
        case_lines = (matrix_output / "cases.jsonl").read_text(
            encoding="utf-8").splitlines()
        matrix_chart = (matrix_output / "matrix.svg").read_text(encoding="utf-8")
        assert matrix["matrix_complete"] is True
        assert matrix["candidate_policy"] == "materialized"
        assert matrix["case_count"] == 8
        assert matrix["all_accuracy_gates_passed"] is True
        assert matrix["all_performance_gates_passed"] is False
        assert matrix["minimum_default_sequence"] == 2048
        assert matrix["minimum_speedup"] == 1.02
        assert matrix["maximum_speedup"] == 1.2
        assert len(case_lines) == 8
        assert "Materialized-score official model boundary" in matrix_chart
    print("cached Attention split model comparison contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
