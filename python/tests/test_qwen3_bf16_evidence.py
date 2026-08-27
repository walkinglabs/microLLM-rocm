#!/usr/bin/env python3
import importlib.util
import json
import math
import tempfile
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
PHASE_ROUTE_ROOT = (ROOT / "benchmarks/results" /
                    "2026-08-27-qwen3-decode-up-fp32-route")
PHASE_GATE_ROOT = (ROOT / "benchmarks/results" /
                   "2026-08-27-qwen3-decode-up-fp32-gate")
BATCH_CONTRACT_ROOT = (ROOT / "benchmarks/results" /
                       "2026-08-27-hf-batch-invariance-contract")
LONG_CONTEXT_ROOT = (ROOT / "benchmarks/results" /
                     "2026-08-27-qwen3-decode-up-fp32-long-context")
PROMPT_PATTERN_ROOT = (ROOT / "benchmarks/results" /
                       "2026-08-27-qwen3-phase-prompt-patterns")
NATURAL_PROMPT_ROOT = (ROOT / "benchmarks/results" /
                       "2026-08-27-qwen3-natural-prompts")
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
PHASE_PERF_SPEC = importlib.util.spec_from_file_location(
    "compare_qwen3_decode_up_fp32_matrix",
    ROOT / "benchmarks/single_gpu/compare_qwen3_decode_up_fp32_matrix.py")
PHASE_PERF = importlib.util.module_from_spec(PHASE_PERF_SPEC)
assert PHASE_PERF_SPEC.loader is not None
PHASE_PERF_SPEC.loader.exec_module(PHASE_PERF)
PHASE_ORACLE_SPEC = importlib.util.spec_from_file_location(
    "qwen3_decode_up_fp32_oracle_sweep",
    ROOT / "benchmarks/single_gpu/qwen3_decode_up_fp32_oracle_sweep.py")
PHASE_ORACLE = importlib.util.module_from_spec(PHASE_ORACLE_SPEC)
assert PHASE_ORACLE_SPEC.loader is not None
PHASE_ORACLE_SPEC.loader.exec_module(PHASE_ORACLE)
PROMPT_MANIFEST_SPEC = importlib.util.spec_from_file_location(
    "make_qwen3_prompt_pattern_manifest",
    ROOT / "benchmarks/single_gpu/make_qwen3_prompt_pattern_manifest.py")
PROMPT_MANIFEST = importlib.util.module_from_spec(PROMPT_MANIFEST_SPEC)
assert PROMPT_MANIFEST_SPEC.loader is not None
PROMPT_MANIFEST_SPEC.loader.exec_module(PROMPT_MANIFEST)
NATURAL_MANIFEST_SPEC = importlib.util.spec_from_file_location(
    "make_qwen3_natural_prompt_manifest",
    ROOT / "benchmarks/single_gpu/make_qwen3_natural_prompt_manifest.py")
NATURAL_MANIFEST = importlib.util.module_from_spec(NATURAL_MANIFEST_SPEC)
assert NATURAL_MANIFEST_SPEC.loader is not None
NATURAL_MANIFEST_SPEC.loader.exec_module(NATURAL_MANIFEST)
EXACT_PROMPT_SPEC = importlib.util.spec_from_file_location(
    "qwen3_exact_prompt_matrix",
    ROOT / "benchmarks/single_gpu/qwen3_exact_prompt_matrix.py")
EXACT_PROMPT = importlib.util.module_from_spec(EXACT_PROMPT_SPEC)
assert EXACT_PROMPT_SPEC.loader is not None
EXACT_PROMPT_SPEC.loader.exec_module(EXACT_PROMPT)

def main():
    natural_tokens = {
        "english": [1, 2], "chinese": [3, 4, 5],
        "code": [6], "chat": [7, 8, 9, 10],
    }
    natural_manifest = NATURAL_MANIFEST.build_manifest({
        "schema_version": 1,
        "models": [{"name": "qwen3-0.6b", "inference": {},
                    "marker": "preserved"}],
    }, natural_tokens)
    assert [model["inference"]["exact_context"]
            for model in natural_manifest["models"]] == [2, 3, 1, 4]
    assert all(model["marker"] == "preserved"
               for model in natural_manifest["models"])
    with tempfile.TemporaryDirectory() as temporary:
        natural_path = Path(temporary) / "natural.json"
        natural_path.write_text(json.dumps(natural_manifest), encoding="utf-8")
        exact_models = EXACT_PROMPT.load_prompts(natural_path)
    exact_summaries = []
    for model in exact_models:
        exact_summaries.append({"rows": [
            {"model": model["name"],
             "context": model["inference"]["exact_context"],
             "status": "pass"} for _ in range(4)]})
    exact_summary = EXACT_PROMPT.aggregate(
        exact_models, exact_summaries, [{"status": "pass"}] * 32)
    assert exact_summary["status"] == "pass"
    assert exact_summary["worker_passes"] == exact_summary["worker_count"] == 32
    assert exact_summary["aggregate_rows"] == exact_summary["pass_rows"] == 16
    prompt_manifest = PROMPT_MANIFEST.build_manifest({
        "schema_version": 1,
        "models": [{"name": "qwen3-0.6b", "inference": {"token_ids": [9]},
                    "marker": "preserved"}],
    })
    assert [model["name"] for model in prompt_manifest["models"]] == [
        "qwen3-constant", "qwen3-alternating", "qwen3-ascending",
        "qwen3-sensitive"]
    assert {model["inference"]["prompt_pattern"]: model["inference"]["token_ids"]
            for model in prompt_manifest["models"]} == PROMPT_MANIFEST.PATTERNS
    assert all(model["marker"] == "preserved"
               for model in prompt_manifest["models"])
    assert len(PHASE_ORACLE.CASES) == 8
    assert PHASE_ORACLE.CANDIDATE == "micro-phase-decode-up-fp32"
    assert len(PHASE_PERF.CASES) == 5
    assert PHASE_PERF.RESIDENT_DELTA_BYTES == 352_321_536
    phase_synthetic = []
    for case in PHASE_PERF.CASES:
        for process_run in range(1, PHASE_PERF.RUNS + 1):
            for policy in PHASE_PERF.POLICIES:
                candidate = policy == "decode-up-fp32"
                resident = 1_503_395_840 + (
                    PHASE_PERF.RESIDENT_DELTA_BYTES if candidate else 0)
                phase_synthetic.append({
                    "case": case["name"], "policy": policy,
                    "throughput_tokens_per_second": 99.0 if candidate else 100.0,
                    "latency_ms": 101.0 if candidate else 100.0,
                    "resident_weight_bytes": resident,
                    "engine_peak_bytes": resident + 4_194_304,
                    "engine_incremental_peak_bytes": 4_194_304,
                    "preparation_peak_bytes": 2_912_681_984,
                    "output_signature": [
                        99 if candidate else 1, case["decode_tokens"]],
                    "process_run": process_run,
                })
    phase_synthetic_summary = PHASE_PERF.summarize(
        phase_synthetic, {"name": "qwen3-0.6b", "revision": "fixture"})
    assert phase_synthetic_summary["status"] == "pass_performance"
    assert math.isclose(
        phase_synthetic_summary[
            "candidate_over_current_throughput_geometric_mean"], 0.99)
    assert all(not case["outputs_equal_across_policies"]
               for case in phase_synthetic_summary["cases"])
    for row in phase_synthetic:
        if row["case"] == "prefill_T512_B2" and \
                row["policy"] == "decode-up-fp32":
            row["throughput_tokens_per_second"] = 90.0
    assert PHASE_PERF.summarize(
        phase_synthetic, {"name": "qwen3-0.6b", "revision": "fixture"})[
            "status"] == "reject_performance"
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
    phase_route = json.loads((PHASE_ROUTE_ROOT / "summary.json").read_text())
    phase_smoke = json.loads((PHASE_ROUTE_ROOT / "smoke.json").read_text())
    assert phase_route["status"] == "pass_route_smoke_unmeasured"
    assert phase_route["route"]["explicit_phase_from_call_path"] is True
    assert phase_route["route"]["sequence_length_inference_used"] is False
    assert phase_route["route"]["mirror_is_parameter"] is False
    assert phase_route["route"]["mirror_is_checkpointed"] is False
    assert phase_route["route"]["default_enabled"] is False
    assert phase_route["official_smoke"][
        "resident_delta_over_all_bf16_bytes"] == 352_321_536
    assert phase_route["tests"] == {
        "cpu_passed": 433, "cpu_total": 433,
        "sanitizer_passed": 430, "sanitizer_total": 430,
        "hip_passed": 215, "hip_total": 215,
        "shape_runner_contract_passed": 82,
        "shape_runner_contract_total": 82,
    }
    assert phase_smoke["status"] == "pass"
    assert phase_smoke["inference_weight_policy"] == \
        "dual_representation_bf16_prefill_decode_up_fp32"
    assert phase_smoke["bf16_ffn_converted_tensors"] == 56
    assert phase_smoke["bf16_ffn_fp32_decode_tensors_retained"] == 28
    assert phase_smoke["bf16_ffn_fp32_decode_bytes_retained"] == 352_321_536
    assert phase_smoke["bf16_ffn_bf16_prefill_mirror_tensors"] == 28
    assert phase_smoke[
        "bf16_ffn_bf16_prefill_mirror_bytes_retained"] == 176_160_768
    assert phase_smoke["resident_weight_bytes"] == 1_855_717_376
    assert phase_smoke["generated_tokens"] == [25]
    phase_gate = json.loads((PHASE_GATE_ROOT / "summary.json").read_text())
    phase_shape = json.loads((PHASE_GATE_ROOT / "shape-summary.json").read_text())
    phase_shape_raw = [json.loads(line) for line in
                       (PHASE_GATE_ROOT / "shape-raw.jsonl").read_text().splitlines()
                       if line]
    phase_oracle = json.loads((PHASE_GATE_ROOT / "oracle-summary.json").read_text())
    phase_oracle_raw = [json.loads(line) for line in
                        (PHASE_GATE_ROOT / "oracle-raw.jsonl").read_text().splitlines()
                        if line]
    phase_performance = json.loads(
        (PHASE_GATE_ROOT / "performance-summary.json").read_text())
    phase_performance_raw = [json.loads(line) for line in
                             (PHASE_GATE_ROOT / "performance-raw.jsonl").read_text().splitlines()
                             if line]
    phase_performance_repeat = json.loads(
        (PHASE_GATE_ROOT / "performance-repeat-summary.json").read_text())
    phase_performance_repeat_raw = [json.loads(line) for line in
        (PHASE_GATE_ROOT / "performance-repeat-raw.jsonl").read_text().splitlines()
        if line]
    assert phase_gate["status"] == "pass_explicit_precision_policy"
    assert phase_gate["shape_gate"]["worker_passes"] == 64
    assert phase_gate["oracle_gate"]["argmax_cases_passed"] == 8
    assert phase_gate["oracle_gate"]["strict_complete_logit_cases_passed"] == 7
    assert phase_gate["performance_gate"]["cases_passed"] == 5
    assert phase_gate["performance_gate"]["independent_matrix_count"] == 2
    assert phase_gate["performance_gate"]["total_process_records"] == 60
    assert phase_gate["performance_gate"]["total_case_gates_passed"] == 10
    assert phase_gate["memory_gate"]["resident_delta_bytes"] == 352_321_536
    assert len(phase_shape["rows"]) == 32 and len(phase_shape_raw) == 64
    assert all(item["status"] == "pass" for item in phase_shape_raw)
    assert sum(item["status"] == "pass" for item in phase_shape["rows"]) == 23
    assert sum(item["status"] == "precision_mismatch"
               for item in phase_shape["rows"]) == 9
    assert "decode_up_fp32=true" in phase_shape[
        "precision_boundary"]["microllm"]
    phase_cached = [item for item in phase_shape["rows"]
                    if item["workload"] == "decode"]
    assert len(phase_cached) == 24
    assert all(item["microllm_kv_cache_actual_bytes"] ==
               item["microllm_kv_cache_theoretical_bytes"] ==
               item["pytorch_kv_cache_actual_bytes"] ==
               item["pytorch_kv_cache_theoretical_bytes"]
               for item in phase_cached)
    assert phase_oracle["status"] == \
        "pass_all_argmax_with_recorded_fp32_alignment_limit"
    assert phase_oracle["oracle_cases_passed"] == len(PHASE_ORACLE.CASES) == 8
    assert phase_oracle["strict_complete_logit_cases_passed"] == 7
    assert phase_oracle["strict_complete_logit_gate"] is False
    assert len(phase_oracle_raw) == 32
    assert all(item["candidate_matches_oracle"] for item in phase_oracle["rows"])
    limited = [item for item in phase_oracle["rows"]
               if not item["fp32_complete_logit_alignment_gate"]]
    assert len(limited) == 1 and limited[0]["name"] == "t128-b1-step8"
    assert limited[0]["micro_fp32_argmax"] == \
        limited[0]["oracle_argmax"] == limited[0]["candidate_argmax"] == 320
    assert len(list((PHASE_GATE_ROOT / "oracle-cases").glob("*.json"))) == 8
    assert phase_performance["status"] == "pass_performance"
    assert len(phase_performance_raw) == 30
    assert all(item["status"] == "pass" for item in phase_performance_raw)
    assert 0.979 < phase_performance[
        "candidate_over_current_throughput_geometric_mean"] < 0.981
    phase_perf_cases = {item["name"]: item
                        for item in phase_performance["cases"]}
    assert all(item["status"] == "pass" for item in phase_perf_cases.values())
    assert phase_perf_cases["prefill_T512_B2"][
        "candidate_over_current_throughput"] > 1.0
    assert all(item["gates"]["incremental_peak_within_tolerance"]
               for item in phase_perf_cases.values())
    assert phase_performance_repeat["status"] == "pass_performance"
    assert len(phase_performance_repeat_raw) == 30
    assert all(item["status"] == "pass" for item in phase_performance_repeat_raw)
    assert 0.981 < phase_performance_repeat[
        "candidate_over_current_throughput_geometric_mean"] < 0.983
    assert all(item["status"] == "pass"
               for item in phase_performance_repeat["cases"])
    batch_contract = json.loads(
        (BATCH_CONTRACT_ROOT / "summary.json").read_text())
    batch_matrix = json.loads(
        (BATCH_CONTRACT_ROOT / "matrix-summary.json").read_text())
    batch_raw = [json.loads(line) for line in
                 (BATCH_CONTRACT_ROOT / "matrix-raw.jsonl").read_text().splitlines()
                 if line]
    batch_oracle = json.loads(
        (BATCH_CONTRACT_ROOT / "oracle-summary.json").read_text())
    assert batch_contract["status"] == "pass_batch_invariance_evidence_contract"
    assert batch_contract["aggregate_status"] == "batch_invariance_mismatch"
    assert batch_contract["microllm"]["generated_rows_equal"] is True
    assert batch_contract["pytorch_bf16"]["generated_rows_equal"] is False
    assert batch_contract["common_fp32_oracle"]["pytorch_fp32_argmax"] == 2
    assert batch_contract["common_fp32_oracle"]["phase_candidate_argmax"] == 2
    assert len(batch_raw) == 2 and len(batch_matrix["rows"]) == 1
    batch_row = batch_matrix["rows"][0]
    assert batch_row["status"] == "batch_invariance_mismatch"
    assert batch_row["microllm_generated_rows"] == \
        batch_contract["microllm"]["generated_rows"]
    assert batch_row["pytorch_generated_rows"] == \
        batch_contract["pytorch_bf16"]["generated_rows"]
    assert batch_oracle["status"] == "pass_diagnosed_precision_policy"
    batch_oracle_rows = {item["policy"]: item
                         for item in batch_oracle["policy_rows"]}
    assert batch_oracle_rows["torch-fp32"]["argmax_token"] == 2
    assert batch_oracle_rows["micro-fp32-fp32"]["argmax_token"] == 2
    assert batch_oracle_rows["micro-phase-decode-up-fp32"]["argmax_token"] == 2
    assert batch_oracle_rows["torch-bf16"]["argmax_token"] == 474
    long_result = json.loads((LONG_CONTEXT_ROOT / "summary.json").read_text())
    long_matrix = json.loads(
        (LONG_CONTEXT_ROOT / "shape-summary.json").read_text())
    long_raw = [json.loads(line) for line in
                (LONG_CONTEXT_ROOT / "shape-raw.jsonl").read_text().splitlines()
                if line]
    long_t1024 = json.loads(
        (LONG_CONTEXT_ROOT / "t1024-b2-step3-oracle-summary.json").read_text())
    long_t2048 = json.loads(
        (LONG_CONTEXT_ROOT / "t2048-b2-step4-oracle-summary.json").read_text())
    assert long_result["status"] == "pass_explicit_policy_long_context_with_limits"
    assert long_result["matrix"]["worker_passes"] == 32
    assert long_result["matrix"]["pass_rows"] == 10
    assert long_result["matrix"]["precision_mismatch_rows"] == 4
    assert long_result["matrix"]["batch_invariance_mismatch_rows"] == 2
    assert long_result["matrix"]["microllm_batch_rows_equal"] == 6
    assert long_result["matrix"]["pytorch_batch_rows_equal"] == 4
    assert long_result["extended_oracle"]["combined_argmax_cases_passed"] == 10
    assert long_result["extended_oracle"][
        "combined_strict_complete_logit_cases_passed"] == 8
    assert len(long_matrix["rows"]) == 16 and len(long_raw) == 32
    assert all(item["status"] == "pass" for item in long_raw)
    assert sum(item["status"] == "pass" for item in long_matrix["rows"]) == 10
    assert sum(item["status"] == "precision_mismatch"
               for item in long_matrix["rows"]) == 4
    assert sum(item["status"] == "batch_invariance_mismatch"
               for item in long_matrix["rows"]) == 2
    long_cached = [item for item in long_matrix["rows"]
                   if item["workload"] == "decode"]
    assert len(long_cached) == 12
    assert all(item["microllm_kv_cache_actual_bytes"] ==
               item["microllm_kv_cache_theoretical_bytes"] ==
               item["pytorch_kv_cache_actual_bytes"] ==
               item["pytorch_kv_cache_theoretical_bytes"]
               for item in long_cached)
    long_b2 = [item for item in long_cached if item["batch"] == 2]
    assert len(long_b2) == 6
    assert sum(item["microllm_generated_rows_equal"] for item in long_b2) == 6
    assert sum(item["pytorch_generated_rows_equal"] for item in long_b2) == 4
    for oracle, candidate_token, torch_token, strict in (
            (long_t1024, 2, 474, True),
            (long_t2048, 16, 220, False)):
        rows = {item["policy"]: item for item in oracle["policy_rows"]}
        assert rows["torch-fp32"]["argmax_token"] == candidate_token
        assert rows["micro-fp32-fp32"]["argmax_token"] == candidate_token
        assert rows["micro-phase-decode-up-fp32"]["argmax_token"] == \
            candidate_token
        assert rows["torch-bf16"]["argmax_token"] == torch_token
        assert oracle["gates"]["fp32_implementations_aligned"] is strict
    long_t1024_rows = {item["policy"]: item
                       for item in long_t1024["policy_rows"]}
    assert long_t1024_rows["micro-phase-decode-up-fp32"][
        "generated_rows_equal"] is True
    assert long_t1024_rows["torch-bf16"]["generated_rows_equal"] is False
    assert long_t1024_rows["torch-bf16"]["generated_rows"][0][-1] == 474
    assert long_t1024_rows["torch-bf16"]["generated_rows"][1][-1] == 2
    pattern_result = json.loads(
        (PROMPT_PATTERN_ROOT / "summary.json").read_text())
    pattern_matrix = json.loads(
        (PROMPT_PATTERN_ROOT / "matrix-summary.json").read_text())
    pattern_raw = [json.loads(line) for line in
                   (PROMPT_PATTERN_ROOT / "matrix-raw.jsonl").read_text().splitlines()
                   if line]
    pattern_seeds = json.loads(
        (PROMPT_PATTERN_ROOT / "prompt-seeds.json").read_text())
    assert pattern_result["status"] == \
        "pass_prompt_pattern_matrix_with_constant_limits"
    assert pattern_result["matrix"]["worker_passes"] == 64
    assert pattern_result["matrix"]["pass_rows"] == 29
    assert pattern_result["matrix"]["precision_mismatch_rows"] == 3
    assert pattern_result["matrix"]["batch_invariance_mismatch_rows"] == 0
    assert pattern_result["matrix"]["microllm_b2_rows_equal"] == 8
    assert pattern_result["matrix"]["pytorch_b2_rows_equal"] == 8
    assert pattern_result["patterns"] == {
        "constant": {"pass_rows": 5, "precision_mismatch_rows": 3},
        "alternating": {"pass_rows": 8, "precision_mismatch_rows": 0},
        "ascending": {"pass_rows": 8, "precision_mismatch_rows": 0},
        "sensitive": {"pass_rows": 8, "precision_mismatch_rows": 0},
    }
    assert pattern_seeds["patterns"] == PROMPT_MANIFEST.PATTERNS
    assert len(pattern_matrix["rows"]) == 32 and len(pattern_raw) == 64
    assert all(item["status"] == "pass" for item in pattern_raw)
    assert sum(item["status"] == "pass" for item in pattern_matrix["rows"]) == 29
    assert sum(item["status"] == "precision_mismatch"
               for item in pattern_matrix["rows"]) == 3
    pattern_cached = [item for item in pattern_matrix["rows"]
                      if item["workload"] == "decode"]
    assert len(pattern_cached) == 16
    assert all(item["microllm_kv_cache_actual_bytes"] ==
               item["microllm_kv_cache_theoretical_bytes"] ==
               item["pytorch_kv_cache_actual_bytes"] ==
               item["pytorch_kv_cache_theoretical_bytes"]
               for item in pattern_cached)
    pattern_b2 = [item for item in pattern_cached if item["batch"] == 2]
    assert len(pattern_b2) == 8
    assert all(item["microllm_generated_rows_equal"] and
               item["pytorch_generated_rows_equal"] for item in pattern_b2)
    mismatches = [item for item in pattern_matrix["rows"]
                  if item["status"] == "precision_mismatch"]
    assert all(item["model"] == "qwen3-constant" for item in mismatches)
    assert {item["model"] for item in pattern_matrix["rows"]
            if item["status"] != "pass"} == {"qwen3-constant"}
    for path, token, torch_token in (
            ("t512-b1-step2.json", 2955, 1096),
            ("t512-b2-step2.json", 2955, 1096),
            ("t2048-b2-step4.json", 16, 220)):
        oracle = json.loads(
            (PROMPT_PATTERN_ROOT / "oracles" / path).read_text())
        rows = {item["policy"]: item for item in oracle["policy_rows"]}
        candidate_name = ("micro-phase-decode-up-fp32"
                          if "micro-phase-decode-up-fp32" in rows
                          else "micro-mixed-gate-down-bf16")
        assert rows["torch-fp32"]["argmax_token"] == token
        assert rows[candidate_name]["argmax_token"] == token
        assert rows["torch-bf16"]["argmax_token"] == torch_token
    natural_result = json.loads(
        (NATURAL_PROMPT_ROOT / "summary.json").read_text())
    natural_matrix = json.loads(
        (NATURAL_PROMPT_ROOT / "matrix-summary.json").read_text())
    natural_raw = [json.loads(line) for line in
                   (NATURAL_PROMPT_ROOT / "matrix-raw.jsonl").read_text().splitlines()
                   if line]
    natural_prompts = json.loads(
        (NATURAL_PROMPT_ROOT / "prompts.json").read_text())
    assert natural_result["status"] == \
        "pass_exact_natural_prompts_with_two_oracle_splits"
    assert natural_result["matrix"]["worker_passes"] == 32
    assert natural_result["matrix"]["pass_rows"] == 14
    assert natural_result["matrix"]["precision_mismatch_rows"] == 2
    assert natural_result["matrix"]["batch_invariance_mismatch_rows"] == 0
    assert natural_result["oracles"]["cases_passed"] == 4
    assert natural_result["oracles"]["strict_complete_logit_cases_passed"] == 4
    assert {item["family"]: len(item["token_ids"])
            for item in natural_prompts["prompts"]} == {
                "english": 22, "chinese": 15, "code": 18, "chat": 24}
    assert len(natural_matrix["rows"]) == 16 and len(natural_raw) == 32
    assert natural_matrix["worker_passes"] == natural_matrix["worker_count"] == 32
    assert natural_matrix["pass_rows"] == 14
    assert natural_matrix["precision_mismatch_rows"] == 2
    assert natural_matrix["batch_invariance_mismatch_rows"] == 0
    natural_cached = [item for item in natural_matrix["rows"]
                      if item["workload"] == "decode"]
    assert len(natural_cached) == 8
    assert all(item["microllm_kv_cache_actual_bytes"] ==
               item["microllm_kv_cache_theoretical_bytes"] ==
               item["pytorch_kv_cache_actual_bytes"] ==
               item["pytorch_kv_cache_theoretical_bytes"]
               for item in natural_cached)
    assert all(item["microllm_generated_rows_equal"] and
               item["pytorch_generated_rows_equal"]
               for item in natural_cached if item["batch"] == 2)
    natural_rows = {(item["prompt_family"], item["batch"], item["workload"]): item
                    for item in natural_matrix["rows"]}
    assert natural_rows[("english", 1, "decode")]["status"] == \
        "precision_mismatch"
    assert natural_rows[("english", 2, "decode")]["status"] == "pass"
    assert natural_rows[("chinese", 1, "decode")]["status"] == "pass"
    assert natural_rows[("chinese", 2, "decode")]["status"] == \
        "precision_mismatch"
    assert all(natural_rows[(family, batch, workload)]["status"] == "pass"
               for family in ("code", "chat")
               for batch in (1, 2) for workload in ("prefill", "decode"))
    for family, step, token, torch_by_batch in (
            ("english", 6, 4416, {1: 785, 2: 4416}),
            ("chinese", 2, 104136, {1: 104136, 2: 3837})):
        for batch in (1, 2):
            oracle = json.loads((NATURAL_PROMPT_ROOT / "oracles" /
                f"{family}-b{batch}-step{step}-summary.json").read_text())
            rows = {item["policy"]: item for item in oracle["policy_rows"]}
            assert oracle["status"] == "pass_diagnosed_precision_policy"
            assert all(oracle["gates"].values())
            assert rows["torch-fp32"]["argmax_token"] == token
            assert rows["micro-fp32-fp32"]["argmax_token"] == token
            assert rows["micro-phase-decode-up-fp32"]["argmax_token"] == token
            assert rows["torch-bf16"]["argmax_token"] == torch_by_batch[batch]
    print("qwen3 bf16 evidence: pass")

if __name__ == "__main__":
    main()
