#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "benchmarks/results/2026-08-26-qwen3-official-alignment/summary.json"
MATRIX = (ROOT / "benchmarks/results" /
          "2026-08-26-qwen3-fixture-shape-matrix/summary.json")
MATRIX_FULL = (ROOT / "benchmarks/results" /
               "2026-08-26-qwen3-fixture-shape-matrix/matrix-summary.json")
MATRIX_RAW = (ROOT / "benchmarks/results" /
              "2026-08-26-qwen3-fixture-shape-matrix/raw.jsonl")


def main():
    row = json.loads(SUMMARY.read_text())
    assert row["status"] == "pass"
    assert row["strict_streaming_loaded_tensors"] == 310
    assert row["stored_tensors"] == 311
    assert row["stored_parameter_values"] == 751_632_384
    assert row["unique_runtime_parameters"] == 596_049_920
    assert row["tied_alias_payload_exact"] is True
    assert row["tied_alias_bytes"] == 311_164_928
    assert row["logit_elements"] == 151_936
    assert row["logit_maximum_absolute_error"] <= 4.0e-5
    assert row["logit_rms_error"] <= 9.0e-6
    assert row["microllm_argmax"] == row["pytorch_argmax"] == 14_582
    assert row["microllm_generated_tokens"] == row["pytorch_generated_tokens"]
    assert row["microllm_generated_tokens"] == [14_582, 25, 16_246, 264]
    assert row["decode_tokens_per_second"] > 0
    matrix = json.loads(MATRIX.read_text())
    assert matrix["status"] == "complete_with_recorded_limits"
    assert matrix["stored_parameter_count"] == 751_632_384
    assert matrix["runtime_parameter_count"] == 596_049_920
    assert matrix["framework_execution_passes"] == \
        matrix["framework_execution_rows"] == 64
    assert matrix["aggregate_shape_rows"] == 32
    assert matrix["cross_framework_pass_rows"] == 24
    assert matrix["precision_mismatch_rows"] == 8
    assert matrix["prefill_top_token_equal_rows"] == matrix["prefill_rows"] == 8
    assert matrix["decode_token_exact_rows"] == 16
    assert matrix["decode_rows"] == matrix["kv_active_bytes_exact_rows"] == 24
    assert len(matrix["mismatches"]) == matrix["precision_mismatch_rows"]
    assert all(item["matching_prefix_tokens"] > 0
               for item in matrix["mismatches"])
    full = json.loads(MATRIX_FULL.read_text())
    raw = [json.loads(line) for line in MATRIX_RAW.read_text().splitlines() if line]
    assert full["status"] == matrix["status"]
    assert len(full["rows"]) == matrix["aggregate_shape_rows"]
    assert sum(item["status"] == "pass" for item in full["rows"]) == \
        matrix["cross_framework_pass_rows"]
    assert sum(item["status"] == "precision_mismatch" for item in full["rows"]) == \
        matrix["precision_mismatch_rows"]
    assert len(raw) == matrix["framework_execution_rows"]
    assert all(item["status"] == "pass" for item in raw)
    assert {item["framework"] for item in raw} == {"microllm", "pytorch"}
    print("qwen3 official evidence: pass")


if __name__ == "__main__":
    main()
