#!/usr/bin/env python3
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "benchmarks/results/2026-08-26-qwen3-bf16-inference/summary.json"
DIVERGENCE_ROOT = (ROOT / "benchmarks/results" /
                   "2026-08-26-qwen3-bf16-first-divergence")
SWEEP_ROOT = (ROOT / "benchmarks/results" /
              "2026-08-26-qwen3-bf16-oracle-sweep")
ISLAND_ROOT = (ROOT / "benchmarks/results" /
               "2026-08-26-qwen3-bf16-t128-weight-islands")
LAYER_ROOT = (ROOT / "benchmarks/results" /
              "2026-08-26-qwen3-bf16-ffn-layer-search")
PROJECTION_ROOT = (ROOT / "benchmarks/results" /
                   "2026-08-26-qwen3-bf16-ffn-projection-search")
CANDIDATE_ROOT = (ROOT / "benchmarks/results" /
                  "2026-08-26-qwen3-ffn0-4-fp32-reject")
GATE_FP32_ROOT = (ROOT / "benchmarks/results" /
                  "2026-08-26-qwen3-bf16-gate-fp32-reject")
CALIBRATION_ROOT = (ROOT / "benchmarks/results" /
                    "2026-08-26-qwen3-bf16-projection-calibration")
DOWN_REJECT_ROOT = (ROOT / "benchmarks/results" /
                    "2026-08-26-qwen3-down-fp32-reject")
UP_REJECT_ROOT = (ROOT / "benchmarks/results" /
                  "2026-08-27-qwen3-up-fp32-reject")
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "audit_qwen3_bf16_divergence",
    ROOT / "benchmarks/single_gpu/audit_qwen3_bf16_divergence.py")
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(RUNNER)
SWEEP_SPEC = importlib.util.spec_from_file_location(
    "qwen3_bf16_oracle_sweep",
    ROOT / "benchmarks/single_gpu/qwen3_bf16_oracle_sweep.py")
SWEEP = importlib.util.module_from_spec(SWEEP_SPEC)
assert SWEEP_SPEC.loader is not None
SWEEP_SPEC.loader.exec_module(SWEEP)
LAYER_SPEC = importlib.util.spec_from_file_location(
    "qwen3_bf16_ffn_layer_search",
    ROOT / "benchmarks/single_gpu/qwen3_bf16_ffn_layer_search.py")
LAYER_SEARCH = importlib.util.module_from_spec(LAYER_SPEC)
assert LAYER_SPEC.loader is not None
LAYER_SPEC.loader.exec_module(LAYER_SEARCH)
UP_PERF_SPEC = importlib.util.spec_from_file_location(
    "compare_qwen3_up_fp32_matrix",
    ROOT / "benchmarks/single_gpu/compare_qwen3_up_fp32_matrix.py")
UP_PERF = importlib.util.module_from_spec(UP_PERF_SPEC)
assert UP_PERF_SPEC.loader is not None
UP_PERF_SPEC.loader.exec_module(UP_PERF)

def main():
    assert len(UP_PERF.CASES) == 5
    assert {(case["context"], case["batch"], case["decode_tokens"])
            for case in UP_PERF.CASES} == {
                (1, 1, 1), (32, 1, 4), (128, 2, 32),
                (512, 2, 0), (512, 2, 32)}
    synthetic = []
    for case in UP_PERF.CASES:
        for process_run in range(1, UP_PERF.RUNS + 1):
            for policy in UP_PERF.POLICIES:
                candidate = policy == "up-fp32"
                synthetic.append({
                    "case": case["name"], "policy": policy,
                    "throughput_tokens_per_second": 98.0 if candidate else 100.0,
                    "latency_ms": 102.0 if candidate else 100.0,
                    "resident_weight_bytes":
                        1_503_395_840 +
                        (UP_PERF.RESIDENT_DELTA_BYTES if candidate else 0),
                    "engine_peak_bytes":
                        1_503_395_840 +
                        (UP_PERF.RESIDENT_DELTA_BYTES if candidate else 0) + 4_194_304,
                    "engine_incremental_peak_bytes": 4_194_304,
                    "preparation_peak_bytes": 2_000_000_000,
                    "output_signature": [case["context"], case["decode_tokens"]],
                    "process_run": process_run,
                })
    synthetic_summary = UP_PERF.summarize(
        synthetic, {"name": "qwen3-0.6b", "revision": "fixture"})
    assert synthetic_summary["status"] == "pass_performance"
    assert math.isclose(
        synthetic_summary["candidate_over_current_throughput_geometric_mean"],
        0.98)
    for row in synthetic:
        if row["policy"] == "up-fp32":
            row["output_signature"] = [999, row["output_signature"][-1]]
    changed_output_summary = UP_PERF.summarize(
        synthetic, {"name": "qwen3-0.6b", "revision": "fixture"})
    assert changed_output_summary["status"] == "pass_performance"
    assert all(not case["outputs_equal_across_policies"]
               for case in changed_output_summary["cases"])
    synthetic[1]["throughput_tokens_per_second"] = 90.0
    synthetic[3]["throughput_tokens_per_second"] = 90.0
    synthetic[5]["throughput_tokens_per_second"] = 90.0
    assert UP_PERF.summarize(
        synthetic, {"name": "qwen3-0.6b", "revision": "fixture"})[
            "status"] == "reject_performance"
    row = json.loads(SUMMARY.read_text())
    assert row["status"] == "pass_explicit_policy"
    assert row["bf16_ffn_linears"] == 84
    assert row["bf16_attention_linears"] == 112
    assert row["qk_norm_dtype"] == "float32"
    assert row["tokens"] == [14582, 25, 16246, 264]
    assert row["fp32_oracle_elements"] == 151_936
    assert row["microllm_bf16_vs_fp32_max"] < row["pytorch_bf16_vs_fp32_max"]
    assert row["microllm_bf16_vs_fp32_rms"] < row["pytorch_bf16_vs_fp32_rms"]
    assert row["microllm_warmup"] == row["pytorch_warmup"] == 2
    assert row["microllm_repetitions"] == row["pytorch_repetitions"] == 5
    assert row["microllm_over_pytorch"] > 1.0
    assert row["bf16_resident_weight_bytes"] < row["fp32_resident_weight_bytes"]
    divergence = json.loads((DIVERGENCE_ROOT / "summary.json").read_text())
    raw = [json.loads(line) for line in
           (DIVERGENCE_ROOT / "raw.jsonl").read_text().splitlines() if line]
    assert divergence["status"] == "pass_diagnosed_precision_policy"
    assert all(divergence["gates"].values())
    assert divergence["context"] == 32
    assert divergence["capture_step"] == 1
    assert divergence["vocabulary_size"] == 151_936
    assert len(raw) == len(divergence["policy_rows"]) == 6
    assert all(item["status"] == "pass" for item in raw)
    policies = {item["policy"]: item for item in divergence["policy_rows"]}
    assert policies["torch-fp32"]["argmax_token"] == 374
    assert policies["micro-fp32-fp32"]["argmax_token"] == 374
    assert policies["micro-bf16-bf16"]["argmax_token"] == 374
    assert policies["torch-bf16"]["argmax_token"] == 323
    assert policies["torch-bf16"]["top1_top2_margin"] == 0.0
    assert policies["torch-bf16"]["top3"][0]["logit"] == \
        policies["torch-bf16"]["top3"][1]["logit"] == 14.1875
    assert policies["micro-bf16-bf16"]["top1_top2_margin"] > 0.03
    assert policies["micro-fp32-fp32"]["versus_torch_fp32_maximum_error"] < 6e-5
    assert policies["micro-fp32-fp32"]["versus_torch_fp32_rms_error"] < 1.3e-5
    assert policies["micro-bf16-bf16"]["versus_torch_fp32_maximum_error"] < \
        policies["torch-bf16"]["versus_torch_fp32_maximum_error"]
    maximum, rms, bitwise = RUNNER.error([1.0, 2.0], [1.0, 3.0])
    assert maximum == 1.0 and math.isclose(rms, math.sqrt(0.5)) and not bitwise
    assert RUNNER.top_tokens([0.0, 3.0, 2.0]) == [1, 2, 0]
    sweep = json.loads((SWEEP_ROOT / "summary.json").read_text())
    sweep_raw = [json.loads(line) for line in
                 (SWEEP_ROOT / "raw.jsonl").read_text().splitlines() if line]
    assert sweep["status"] == "pass_all_mismatches_attributed"
    assert sweep["unique_oracle_cases"] == len(SWEEP.CASES) == 5
    assert sweep["matrix_mismatch_rows"] == 8
    assert sweep["micro_oracle_case_wins"] == 4
    assert sweep["torch_oracle_case_wins"] == 1
    assert sweep["micro_oracle_matrix_rows"] == 7
    assert sweep["torch_oracle_matrix_rows"] == 1
    case_rows = {item["name"]: item for item in sweep["case_rows"]}
    expected = {
        "t32-b1-step1": (374, 374, 323),
        "t32-b2-step1": (374, 374, 323),
        "t128-b2-step8": (320, 25, 320),
        "t512-b1-step2": (2955, 2955, 1096),
        "t512-b2-step8-forced": (1273, 1273, 4285),
    }
    assert {name: (item["oracle_argmax"], item["micro_mixed_argmax"],
                   item["torch_bf16_argmax"])
            for name, item in case_rows.items()} == expected
    forced = case_rows["t512-b2-step8-forced"]["forced_inputs"]
    assert forced == [14582, 198, 262, 1096, 374, 279, 2038, 374, 264]
    assert len(sweep_raw) == 28 and all(item["status"] == "pass"
                                        for item in sweep_raw)
    forced_raw = [item for item in sweep_raw
                  if item["oracle_case"] == "t512-b2-step8-forced"]
    assert len(forced_raw) == 4
    assert all(item["forced_decode_inputs"] is True and
               item["forced_decode_input_count"] == 9 for item in forced_raw)
    islands = json.loads((ISLAND_ROOT / "summary.json").read_text())
    island_raw = [json.loads(line) for line in
                  (ISLAND_ROOT / "raw.jsonl").read_text().splitlines() if line]
    assert islands["status"] == "pass_diagnosed_precision_policy"
    assert all(islands["gates"].values())
    assert islands["context"] == 128 and islands["batch"] == 2
    assert islands["forced_inputs"] == \
        [14582, 1, 374, 264, 3491, 429, 374, 537, 264]
    policies = {item["policy"]: item for item in islands["policy_rows"]}
    assert len(policies) == len(island_raw) == 7
    assert policies["micro-fp32-fp32"]["argmax_token"] == 320
    assert policies["micro-ffn-bf16-fp32"]["argmax_token"] == 25
    assert policies["micro-attention-bf16-fp32"]["argmax_token"] == 320
    assert policies["micro-bf16-fp32"]["argmax_token"] == 25
    assert policies["micro-bf16-bf16"]["argmax_token"] == 25
    assert policies["torch-bf16"]["argmax_token"] == 320
    assert policies["micro-ffn-bf16-fp32"]["captured_rows_maximum_error"] > 0.14
    assert policies["micro-attention-bf16-fp32"]["captured_rows_maximum_error"] < 0.033
    raw_by_policy = {item["framework_policy"]: item for item in island_raw}
    assert raw_by_policy["micro-ffn-bf16-fp32"]["bf16_ffn_weights"] is True
    assert raw_by_policy["micro-ffn-bf16-fp32"]["bf16_attention_weights"] is False
    assert raw_by_policy["micro-attention-bf16-fp32"]["bf16_ffn_weights"] is False
    assert raw_by_policy["micro-attention-bf16-fp32"]["bf16_attention_weights"] is True
    layers = json.loads((LAYER_ROOT / "summary.json").read_text())
    layer_raw = [json.loads(line) for line in
                 (LAYER_ROOT / "raw.jsonl").read_text().splitlines() if line]
    assert layers["status"] == "pass_minimal_combinations_found"
    assert all(layers["gates"].values())
    assert layers["process_rows"] == len(layer_raw) == 28
    assert layers["single_layer_flips"] == []
    assert layers["pair_layer_flips"] == [[3, 4]]
    assert layers["minimal_flipping_sets"] == [[0, 1, 2], [3, 4]]
    repeats = {item["name"]: item for item in layers["repeat_rows"]}
    assert repeats["active-0-2"]["argmax_tokens"] == [25, 25, 25]
    assert repeats["pair-3-4"]["argmax_tokens"] == [25, 25, 25]
    assert repeats["pair-4-6"]["argmax_tokens"] == [320, 320, 320]
    assert repeats["pair-4-6"]["minimum_margin"] < 0.00032
    assert len(LAYER_SEARCH.SINGLES) == 7 and len(LAYER_SEARCH.PAIRS) == 9
    assert all(item["converted_tensors"] ==
               len(item["active_bf16_layers"]) * 3 for item in layer_raw)
    assert all(set(item["active_bf16_layers"]).isdisjoint(item["fp32_layers"])
               and len(item["active_bf16_layers"]) + len(item["fp32_layers"]) == 28
               for item in layer_raw)
    projections = json.loads((PROJECTION_ROOT / "summary.json").read_text())
    projection_raw = [json.loads(line) for line in
                      (PROJECTION_ROOT / "raw.jsonl").read_text().splitlines()
                      if line]
    assert projections["status"] == "pass_all_three_projections_required"
    assert all(projections["gates"].values())
    assert projections["case_count"] == 2
    assert projections["scope_rows"] == 12
    assert len(projection_raw) == 20
    assert all(case["all_projection_argmax"] == 25
               for case in projections["cases"])
    assert all(row["argmax_token"] == 320
               for case in projections["cases"] for row in case["scope_rows"])
    expected_scopes = {
        "gate-only", "up-only", "down-only",
        "gate-up", "gate-down", "up-down",
    }
    for case in ("layers-0-1-2", "layers-3-4"):
        raw_scopes = {
            item["bf16_ffn_weight_scope"] for item in projection_raw
            if item["projection_case"] == case and
            item["framework_policy"].startswith("micro-ffn-") and
            item["bf16_ffn_weight_scope"] != "all"
        }
        assert raw_scopes == expected_scopes
    candidate = json.loads((CANDIDATE_ROOT / "summary.json").read_text())
    candidate_matrix = json.loads(
        (CANDIDATE_ROOT / "matrix-summary.json").read_text())
    candidate_raw = [json.loads(line) for line in
                     (CANDIDATE_ROOT / "raw.jsonl").read_text().splitlines()
                     if line]
    assert candidate["status"] == "reject_precision_and_batch_invariance"
    assert candidate["resident_weight_delta_bytes"] == 94_371_840
    assert candidate["performance_gate_run"] is False
    assert candidate["t128_b2_n32"]["current_matching_prefix_tokens"] == 8
    assert candidate["t128_b2_n32"]["candidate_matching_prefix_tokens"] == 22
    assert candidate["t512_b2_n32"]["repeated_failures"] == 3
    assert candidate_matrix["status"] == "complete_with_recorded_limits"
    assert len(candidate_matrix["rows"]) == 32
    assert sum(row["status"] == "pass" for row in candidate_matrix["rows"]) == 23
    assert sum(row["status"] == "precision_mismatch"
               for row in candidate_matrix["rows"]) == 8
    assert sum(row["status"] == "limited" for row in candidate_matrix["rows"]) == 1
    assert len(candidate_raw) == 64
    failures = [row for row in candidate_raw if row["status"] != "pass"]
    assert len(failures) == 1
    assert failures[0]["context"] == 512 and failures[0]["batch"] == 2
    assert "identical batch rows" in failures[0]["error"]
    assert "ffn_fp32_layers=0,1,2,3,4" in \
        candidate_matrix["precision_boundary"]["microllm"]
    gate_fp32 = json.loads((GATE_FP32_ROOT / "summary.json").read_text())
    gate_fp32_raw = [json.loads(line) for line in
                     (GATE_FP32_ROOT / "raw.jsonl").read_text().splitlines()
                     if line]
    assert gate_fp32["status"] == "reject_oracle"
    assert gate_fp32["case_count"] == 5
    assert gate_fp32["gates"]["all_five_cases_present"] is True
    assert gate_fp32["gates"]["candidate_matches_every_fp32_argmax"] is False
    rows = {item["name"]: item for item in gate_fp32["rows"]}
    assert sum(item["candidate_matches_oracle"] for item in rows.values()) == 4
    assert (rows["t512-b1-step2"]["oracle_argmax"],
            rows["t512-b1-step2"]["candidate_argmax"],
            rows["t512-b1-step2"]["torch_bf16_argmax"]) == (2955, 1096, 1096)
    assert rows["t512-b1-step2"]["candidate_margin"] < 0.0033
    assert len(gate_fp32_raw) == 20
    assert all(item["status"] == "pass" for item in gate_fp32_raw)
    calibration = json.loads((CALIBRATION_ROOT / "summary.json").read_text())
    calibration_raw = [json.loads(line) for line in
                       (CALIBRATION_ROOT / "raw.jsonl").read_text().splitlines()
                       if line]
    assert calibration["status"] == "down_fp32_selected_for_shape_gate"
    assert calibration["selected_runner_policy"] == "micro-mixed-gate-up-bf16"
    assert calibration["resident_weight_delta_bytes"] == 176_160_768
    assert calibration["candidate_ffn_bf16_tensors"] == 56
    assert calibration["candidate_attention_bf16_tensors"] == 112
    policies = {item["name"]: item for item in calibration["policies"]}
    assert policies["gate-fp32"]["oracle_cases_passed"] == 4
    assert policies["up-fp32"]["oracle_cases_passed"] == 5
    assert policies["down-fp32"]["oracle_cases_passed"] == 5
    assert policies["down-fp32"]["minimum_top1_top2_margin"] > \
        policies["up-fp32"]["minimum_top1_top2_margin"]
    assert calibration["full_shape_gate_complete"] is False
    assert calibration["performance_gate_complete"] is False
    assert len(calibration_raw) == 40
    for label, policy, scope in (
            ("up-fp32", "micro-mixed-gate-down-bf16", "gate-down"),
            ("down-fp32", "micro-mixed-gate-up-bf16", "gate-up")):
        samples = [item for item in calibration_raw
                   if item["calibration_policy"] == label and
                   item["framework_policy"] == policy]
        assert len(samples) == 5
        assert all(item["bf16_ffn_weight_scope"] == scope and
                   item["resident_weight_bytes"] == 1_679_556_608
                   for item in samples)
    down_reject = json.loads((DOWN_REJECT_ROOT / "summary.json").read_text())
    down_matrix = json.loads(
        (DOWN_REJECT_ROOT / "matrix-summary.json").read_text())
    down_matrix_raw = [json.loads(line) for line in
                       (DOWN_REJECT_ROOT / "matrix-raw.jsonl").read_text().splitlines()
                       if line]
    down_perf = json.loads(
        (DOWN_REJECT_ROOT / "short-performance-summary.json").read_text())
    down_oracle = json.loads(
        (DOWN_REJECT_ROOT / "down-new-oracle-summary.json").read_text())
    up_oracle = json.loads(
        (DOWN_REJECT_ROOT / "up-new-oracle-summary.json").read_text())
    assert down_reject["status"] == "reject_extended_oracle"
    assert down_reject["complete_shape_worker_passes"] == 64
    assert down_reject["complete_shape_pass_rows"] == 22
    assert down_reject["complete_shape_precision_mismatch_rows"] == 10
    assert down_reject["new_t128_b1_step8"]["fp32_argmax"] == 320
    assert down_reject["new_t128_b1_step8"]["down_fp32_argmax"] == 25
    assert down_reject["new_t128_b1_step8"]["up_fp32_argmax"] == 320
    assert down_reject["up_fp32_extended_oracle_cases_passed"] == 6
    assert len(down_matrix["rows"]) == 32 and len(down_matrix_raw) == 64
    assert all(item["status"] == "pass" for item in down_matrix_raw)
    assert sum(item["status"] == "precision_mismatch"
               for item in down_matrix["rows"]) == 10
    assert down_perf["status"] == "pass_performance"
    assert 0.95 <= down_perf["candidate_over_current_throughput"] < 0.96
    down_policies = {item["policy"]: item for item in down_oracle["policy_rows"]}
    up_policies = {item["policy"]: item for item in up_oracle["policy_rows"]}
    assert down_policies["micro-mixed-gate-up-bf16"]["argmax_token"] == 25
    assert up_policies["micro-mixed-gate-down-bf16"]["argmax_token"] == 320
    up_reject = json.loads((UP_REJECT_ROOT / "summary.json").read_text())
    up_matrix = json.loads((UP_REJECT_ROOT / "shape-summary.json").read_text())
    up_matrix_raw = [json.loads(line) for line in
                     (UP_REJECT_ROOT / "shape-raw.jsonl").read_text().splitlines()
                     if line]
    up_perf = json.loads((UP_REJECT_ROOT / "performance-summary.json").read_text())
    up_perf_raw = [json.loads(line) for line in
                   (UP_REJECT_ROOT / "performance-raw.jsonl").read_text().splitlines()
                   if line]
    up_t128 = json.loads(
        (UP_REJECT_ROOT / "t128-b2-step22-oracle-summary.json").read_text())
    up_t512 = json.loads(
        (UP_REJECT_ROOT / "t512-b2-step2-oracle-summary.json").read_text())
    assert up_reject["status"] == "reject_global_performance"
    assert up_reject["shape_gate"]["worker_passes"] == 64
    assert up_reject["shape_gate"]["precision_mismatch_rows"] == 9
    assert up_reject["oracle_gate"]["unique_states_passed"] == 8
    assert up_reject["oracle_gate"]["mismatch_rows_attributed"] == 9
    assert up_reject["performance_gate"]["decode_cases_passed"] == 4
    assert up_reject["performance_gate"]["resident_delta_bytes"] == 176_160_768
    assert up_matrix["status"] == "complete_with_recorded_limits"
    assert len(up_matrix["rows"]) == 32 and len(up_matrix_raw) == 64
    assert all(item["status"] == "pass" for item in up_matrix_raw)
    assert sum(item["status"] == "pass" for item in up_matrix["rows"]) == 23
    assert sum(item["status"] == "precision_mismatch"
               for item in up_matrix["rows"]) == 9
    cached_rows = [item for item in up_matrix["rows"]
                   if item["workload"] == "decode"]
    assert len(cached_rows) == 24
    assert all(item["microllm_kv_cache_actual_bytes"] ==
               item["microllm_kv_cache_theoretical_bytes"] ==
               item["pytorch_kv_cache_actual_bytes"] ==
               item["pytorch_kv_cache_theoretical_bytes"]
               for item in cached_rows)
    assert up_perf["status"] == "reject_performance"
    assert len(up_perf_raw) == 30
    assert all(item["status"] == "pass" for item in up_perf_raw)
    assert 0.957 < up_perf[
        "candidate_over_current_throughput_geometric_mean"] < 0.959
    perf_cases = {item["name"]: item for item in up_perf["cases"]}
    assert sum(item["status"] == "pass" for item in perf_cases.values()) == 4
    assert perf_cases["prefill_T512_B2"]["status"] == "reject"
    assert perf_cases["prefill_T512_B2"][
        "candidate_over_current_throughput"] < 0.89
    assert perf_cases["prefill_T512_B2"][
        "candidate_over_current_latency"] > 1.12
    assert all(item["gates"]["candidate_output_deterministic"] and
               item["gates"]["current_output_deterministic"]
               for item in perf_cases.values())
    for oracle, token, torch_bf16 in ((up_t128, 4226, 3270),
                                      (up_t512, 2955, 1096)):
        assert oracle["status"] == "pass_diagnosed_precision_policy"
        rows = {item["policy"]: item for item in oracle["policy_rows"]}
        assert rows["torch-fp32"]["argmax_token"] == token
        assert rows["micro-fp32-fp32"]["argmax_token"] == token
        assert rows["micro-mixed-gate-down-bf16"]["argmax_token"] == token
        assert rows["torch-bf16"]["argmax_token"] == torch_bf16
    print("qwen3 bf16 evidence: pass")

if __name__ == "__main__":
    main()
