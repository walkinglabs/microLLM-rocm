import importlib.util
import json
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "hf_inference_shape_matrix",
    ROOT / "benchmarks/single_gpu/hf_inference_shape_matrix.py")
MATRIX = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MATRIX)
RENDER_SPEC = importlib.util.spec_from_file_location(
    "render_serving_batch_scale",
    ROOT / "benchmarks/single_gpu/render_serving_batch_scale.py")
RENDER = importlib.util.module_from_spec(RENDER_SPEC)
assert RENDER_SPEC.loader is not None
RENDER_SPEC.loader.exec_module(RENDER)
CROSS_BATCH_SPEC = importlib.util.spec_from_file_location(
    "audit_cached_cross_batch_logits",
    ROOT / "benchmarks/single_gpu/audit_cached_cross_batch_logits.py")
CROSS_BATCH = importlib.util.module_from_spec(CROSS_BATCH_SPEC)
assert CROSS_BATCH_SPEC.loader is not None
CROSS_BATCH_SPEC.loader.exec_module(CROSS_BATCH)
PRECISION_SPEC = importlib.util.spec_from_file_location(
    "audit_cross_batch_precision",
    ROOT / "benchmarks/single_gpu/audit_cross_batch_precision.py")
PRECISION = importlib.util.module_from_spec(PRECISION_SPEC)
assert PRECISION_SPEC.loader is not None
PRECISION_SPEC.loader.exec_module(PRECISION)
BLOCK_DRIFT_SPEC = importlib.util.spec_from_file_location(
    "audit_cached_block_drift",
    ROOT / "benchmarks/single_gpu/audit_cached_block_drift.py")
BLOCK_DRIFT = importlib.util.module_from_spec(BLOCK_DRIFT_SPEC)
assert BLOCK_DRIFT_SPEC.loader is not None
BLOCK_DRIFT_SPEC.loader.exec_module(BLOCK_DRIFT)
BLOCK_DETAIL_SPEC = importlib.util.spec_from_file_location(
    "audit_cached_block_detail",
    ROOT / "benchmarks/single_gpu/audit_cached_block_detail.py")
BLOCK_DETAIL = importlib.util.module_from_spec(BLOCK_DETAIL_SPEC)
assert BLOCK_DETAIL_SPEC.loader is not None
BLOCK_DETAIL_SPEC.loader.exec_module(BLOCK_DETAIL)
LAYER_COUNTERFACTUAL_SPEC = importlib.util.spec_from_file_location(
    "audit_bf16_ffn_layer_counterfactual",
    ROOT / "benchmarks/single_gpu/audit_bf16_ffn_layer_counterfactual.py")
LAYER_COUNTERFACTUAL = importlib.util.module_from_spec(LAYER_COUNTERFACTUAL_SPEC)
assert LAYER_COUNTERFACTUAL_SPEC.loader is not None
LAYER_COUNTERFACTUAL_SPEC.loader.exec_module(LAYER_COUNTERFACTUAL)
DECODE_ALGORITHM_SPEC = importlib.util.spec_from_file_location(
    "audit_bf16_decode_algorithm",
    ROOT / "benchmarks/single_gpu/audit_bf16_decode_algorithm.py")
DECODE_ALGORITHM = importlib.util.module_from_spec(DECODE_ALGORITHM_SPEC)
assert DECODE_ALGORITHM_SPEC.loader is not None
DECODE_ALGORITHM_SPEC.loader.exec_module(DECODE_ALGORITHM)
ROW_INVARIANCE_SPEC = importlib.util.spec_from_file_location(
    "bf16_row_invariance_matrix",
    ROOT / "benchmarks/single_gpu/bf16_row_invariance_matrix.py")
ROW_INVARIANCE = importlib.util.module_from_spec(ROW_INVARIANCE_SPEC)
assert ROW_INVARIANCE_SPEC.loader is not None
ROW_INVARIANCE_SPEC.loader.exec_module(ROW_INVARIANCE)
PREFILL_CACHE_SPEC = importlib.util.spec_from_file_location(
    "audit_prefill_cache_prefix",
    ROOT / "benchmarks/single_gpu/audit_prefill_cache_prefix.py")
PREFILL_CACHE = importlib.util.module_from_spec(PREFILL_CACHE_SPEC)
assert PREFILL_CACHE_SPEC.loader is not None
PREFILL_CACHE_SPEC.loader.exec_module(PREFILL_CACHE)
PREFILL_TRACE_SPEC = importlib.util.spec_from_file_location(
    "audit_prefill_block0_trace",
    ROOT / "benchmarks/single_gpu/audit_prefill_block0_trace.py")
PREFILL_TRACE = importlib.util.module_from_spec(PREFILL_TRACE_SPEC)
assert PREFILL_TRACE_SPEC.loader is not None
PREFILL_TRACE_SPEC.loader.exec_module(PREFILL_TRACE)
FP32_QKV_SPEC = importlib.util.spec_from_file_location(
    "fp32_qkv_row_invariance_matrix",
    ROOT / "benchmarks/single_gpu/fp32_qkv_row_invariance_matrix.py")
FP32_QKV = importlib.util.module_from_spec(FP32_QKV_SPEC)
assert FP32_QKV_SPEC.loader is not None
FP32_QKV_SPEC.loader.exec_module(FP32_QKV)
FP32_QKV_MODEL_SPEC = importlib.util.spec_from_file_location(
    "fp32_qkv_model_gate",
    ROOT / "benchmarks/single_gpu/fp32_qkv_model_gate.py")
FP32_QKV_MODEL = importlib.util.module_from_spec(FP32_QKV_MODEL_SPEC)
assert FP32_QKV_MODEL_SPEC.loader is not None
FP32_QKV_MODEL_SPEC.loader.exec_module(FP32_QKV_MODEL)
POST_CACHE_SPEC = importlib.util.spec_from_file_location(
    "audit_post_cache_block0_trace",
    ROOT / "benchmarks/single_gpu/audit_post_cache_block0_trace.py")
POST_CACHE = importlib.util.module_from_spec(POST_CACHE_SPEC)
assert POST_CACHE_SPEC.loader is not None
POST_CACHE_SPEC.loader.exec_module(POST_CACHE)
ATTENTION_CORE_SPEC = importlib.util.spec_from_file_location(
    "audit_prefill_attention_core",
    ROOT / "benchmarks/single_gpu/audit_prefill_attention_core.py")
ATTENTION_CORE = importlib.util.module_from_spec(ATTENTION_CORE_SPEC)
assert ATTENTION_CORE_SPEC.loader is not None
ATTENTION_CORE_SPEC.loader.exec_module(ATTENTION_CORE)
ATTENTION_SOLUTIONS_SPEC = importlib.util.spec_from_file_location(
    "fp32_attention_batch_invariance_matrix",
    ROOT / "benchmarks/single_gpu/fp32_attention_batch_invariance_matrix.py")
ATTENTION_SOLUTIONS = importlib.util.module_from_spec(ATTENTION_SOLUTIONS_SPEC)
assert ATTENTION_SOLUTIONS_SPEC.loader is not None
ATTENTION_SOLUTIONS_SPEC.loader.exec_module(ATTENTION_SOLUTIONS)
ATTENTION_MODEL_SPEC = importlib.util.spec_from_file_location(
    "fp32_prefill_attention_model_gate",
    ROOT / "benchmarks/single_gpu/fp32_prefill_attention_model_gate.py")
ATTENTION_MODEL = importlib.util.module_from_spec(ATTENTION_MODEL_SPEC)
assert ATTENTION_MODEL_SPEC.loader is not None
ATTENTION_MODEL_SPEC.loader.exec_module(ATTENTION_MODEL)
ATTENTION_SELECTIVE_SPEC = importlib.util.spec_from_file_location(
    "fp32_prefill_attention_selective_gate",
    ROOT / "benchmarks/single_gpu/fp32_prefill_attention_selective_gate.py")
ATTENTION_SELECTIVE = importlib.util.module_from_spec(ATTENTION_SELECTIVE_SPEC)
assert ATTENTION_SELECTIVE_SPEC.loader is not None
ATTENTION_SELECTIVE_SPEC.loader.exec_module(ATTENTION_SELECTIVE)
POST_EXACT_CORE_SPEC = importlib.util.spec_from_file_location(
    "audit_post_exact_core_block0_trace",
    ROOT / "benchmarks/single_gpu/audit_post_exact_core_block0_trace.py")
POST_EXACT_CORE = importlib.util.module_from_spec(POST_EXACT_CORE_SPEC)
assert POST_EXACT_CORE_SPEC.loader is not None
POST_EXACT_CORE_SPEC.loader.exec_module(POST_EXACT_CORE)
POST_EXACT_O_SPEC = importlib.util.spec_from_file_location(
    "audit_post_exact_o_block0_trace",
    ROOT / "benchmarks/single_gpu/audit_post_exact_o_block0_trace.py")
POST_EXACT_O = importlib.util.module_from_spec(POST_EXACT_O_SPEC)
assert POST_EXACT_O_SPEC.loader is not None
POST_EXACT_O_SPEC.loader.exec_module(POST_EXACT_O)
O_MODEL_SPEC = importlib.util.spec_from_file_location(
    "fp32_prefill_o_model_gate",
    ROOT / "benchmarks/single_gpu/fp32_prefill_o_model_gate.py")
O_MODEL = importlib.util.module_from_spec(O_MODEL_SPEC)
assert O_MODEL_SPEC.loader is not None
O_MODEL_SPEC.loader.exec_module(O_MODEL)
EXACT_STACK_SPEC = importlib.util.spec_from_file_location(
    "fp32_prefill_exact_stack_gate",
    ROOT / "benchmarks/single_gpu/fp32_prefill_exact_stack_gate.py")
EXACT_STACK = importlib.util.module_from_spec(EXACT_STACK_SPEC)
assert EXACT_STACK_SPEC.loader is not None
EXACT_STACK_SPEC.loader.exec_module(EXACT_STACK)
FFN_STAGES_SPEC = importlib.util.spec_from_file_location(
    "audit_prefill_ffn_stages",
    ROOT / "benchmarks/single_gpu/audit_prefill_ffn_stages.py")
FFN_STAGES = importlib.util.module_from_spec(FFN_STAGES_SPEC)
assert FFN_STAGES_SPEC.loader is not None
FFN_STAGES_SPEC.loader.exec_module(FFN_STAGES)
FFN_SOLUTIONS_SPEC = importlib.util.spec_from_file_location(
    "fp32_ffn_row_invariance_matrix",
    ROOT / "benchmarks/single_gpu/fp32_ffn_row_invariance_matrix.py")
FFN_SOLUTIONS = importlib.util.module_from_spec(FFN_SOLUTIONS_SPEC)
assert FFN_SOLUTIONS_SPEC.loader is not None
FFN_SOLUTIONS_SPEC.loader.exec_module(FFN_SOLUTIONS)
FFN_DOWN_SPEC = importlib.util.spec_from_file_location(
    "fp32_ffn_down_row_invariance",
    ROOT / "benchmarks/single_gpu/fp32_ffn_down_row_invariance.py")
FFN_DOWN = importlib.util.module_from_spec(FFN_DOWN_SPEC)
assert FFN_DOWN_SPEC.loader is not None
FFN_DOWN_SPEC.loader.exec_module(FFN_DOWN)
NATIVE128_SPEC = importlib.util.spec_from_file_location(
    "native128_finalize_matrix",
    ROOT / "benchmarks/single_gpu/native128_finalize_matrix.py")
NATIVE128 = importlib.util.module_from_spec(NATIVE128_SPEC)
assert NATIVE128_SPEC.loader is not None
NATIVE128_SPEC.loader.exec_module(NATIVE128)


class HfInferenceShapeMatrixTest(unittest.TestCase):
    def test_native128_summary_requires_every_t2048_case(self):
        rows = []
        for sequence in NATIVE128.SEQUENCES:
            for batch in NATIVE128.BATCHES:
                for dtype in NATIVE128.DTYPES:
                    for run in (1, 2):
                        speed = 1.06
                        if sequence == 2048 and batch == 2 and dtype == "bf16":
                            speed = 1.01
                        rows.append({
                            "sequence": sequence, "batch": batch,
                            "cache_dtype": dtype, "process_run": run,
                            "native128_max_error": 1.0e-6,
                            "native128_rms_error": 1.0e-7,
                            "native128_bitwise_equal_materialized": False,
                            "native128_event_speedup": speed,
                            "native128_wall_speedup": speed,
                            "native128_backend_allocation_calls_per_invocation": 0,
                        })
        summary = NATIVE128.summarize(rows)
        self.assertEqual(summary["process_rows"], 16)
        self.assertEqual(summary["case_count"], 8)
        self.assertEqual(summary["t2048_performance_pass_count"], 3)
        self.assertFalse(summary["candidate_admitted"])
        ET.fromstring(NATIVE128.render(summary))

    def test_current_finalize_gap_selects_native_not_logical_128(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-finalize-architecture-gap-audit")
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(analysis["current_physical_threads"], 256)
        self.assertEqual(analysis["deepseek_width"], 128)
        self.assertEqual(analysis["idle_threads_during_pv"], 128)
        self.assertEqual(
            analysis["rejected_exact_mapping_preserved_logical_lanes"], 256)
        self.assertIn("native 128-lane", analysis["selected_hypothesis"])
        self.assertTrue(analysis["numerical_order_changes"])
        ET.parse(root / "finalize-gap.svg")

    def test_current_clean_long_context_baseline_and_profile(self):
        baseline_root = (ROOT / "benchmarks/results" /
                         "2026-08-26-clean-deepseek-t2048")
        baseline = json.loads((baseline_root / "summary.json").read_text(
            encoding="utf-8"))
        analysis = json.loads((baseline_root / "analysis.json").read_text(
            encoding="utf-8"))
        verification = json.loads((baseline_root / "verification.json").read_text(
            encoding="utf-8"))
        raw = [json.loads(line) for line in
               (baseline_root / "raw.jsonl").read_text(
                   encoding="utf-8").splitlines() if line]
        self.assertEqual(len(raw), 6)
        self.assertEqual(baseline["runs_per_framework"], 3)
        row = baseline["rows"][0]
        self.assertTrue(row["cross_framework_tokens_equal"])
        self.assertEqual(row["cross_framework_matching_prefix_tokens"], 64)
        self.assertAlmostEqual(row["throughput_ratio_microllm_over_pytorch"],
                               1.139273731551338)
        self.assertEqual(row["microllm_peak_bytes"], 5229860864.0)
        self.assertEqual(row["pytorch_peak_bytes"], 6381346816.0)
        self.assertEqual(row["microllm_kv_cache_actual_bytes"], 121110528.0)
        self.assertEqual(row["pytorch_kv_cache_actual_bytes"], 121110528.0)
        self.assertEqual(analysis["matching_generated_tokens"], 64)
        self.assertEqual(verification["runs_per_framework"], 3)

        profile_root = (ROOT / "benchmarks/results" /
                        "2026-08-26-clean-deepseek-t2048-profile")
        profile = json.loads((profile_root / "summary.json").read_text(
            encoding="utf-8"))
        profile_analysis = json.loads((profile_root / "analysis.json").read_text(
            encoding="utf-8"))
        profile_verification = json.loads((profile_root / "verification.json").read_text(
            encoding="utf-8"))
        categories = {item["category"]: item for item in
                      profile["kernel_profile"]["categories"]}
        self.assertAlmostEqual(
            categories["cached Attention finalize"]["kernel_share"],
            0.42269118556170254)
        self.assertAlmostEqual(categories["hipBLASLt GEMM"]["kernel_share"],
                               0.3325377735483638)
        self.assertEqual(profile["kernel_profile"][
            "negative_call_delta_names"], [])
        self.assertEqual(profile_analysis["largest_category"],
                         "cached Attention finalize")
        self.assertEqual(profile_verification["derived_forward_steps"], 128)
        ET.parse(profile_root / "profile-delta.svg")

    def test_current_ffn_down_solution_matrix_rejects_exact_candidate(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-fp32-ffn-down-row-invariance")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        raw = [json.loads(line) for line in
               (root / "raw.jsonl").read_text(encoding="utf-8").splitlines()
               if line]
        self.assertEqual(summary["record_type"],
                         "fp32_ffn_down_row_invariance_matrix")
        self.assertEqual((summary["inner"], summary["columns"]), (8960, 1536))
        self.assertEqual((len(raw), summary["common_candidate_count"]), (15, 15))
        self.assertEqual(summary["block_invariant_indices"], [296100])
        self.assertEqual(summary["performance_admitted_count"], 0)
        self.assertEqual(summary["recommended_index"], -1)
        exact = next(row for row in summary["candidates"]
                     if row["index"] == 296100)
        self.assertEqual(exact["speedup_vs_default"], [
            0.506474504, 0.758019906, 0.6860566, 0.862578057])
        self.assertAlmostEqual(exact["minimum_speedup"], 0.506474504)
        self.assertFalse(analysis["model_gate_required"])
        self.assertTrue(verification["focused_gates"][
            "rejected_gate_up_route_absent"])
        ET.parse(root / "ffn-down-row-invariance.svg")

    def test_ffn_down_runner_uses_clean_real_descriptor(self):
        self.assertEqual(FFN_DOWN.BASE.ROWS, [2048, 4096, 8192, 16384])
        self.assertEqual(FFN_DOWN.BASE.INNER, 8960)
        self.assertEqual(FFN_DOWN.BASE.COLUMNS, 1536)
        self.assertFalse((ROOT / "benchmarks/single_gpu" /
                          "fp32_prefill_ffn_model_gate.py").exists())

    def test_current_post_exact_gate_up_trace_selects_down(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-post-exact-gate-up-ffn-trace")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        raw = [json.loads(line) for line in
               (root / "raw.jsonl").read_text(encoding="utf-8").splitlines()
               if line]
        self.assertEqual(summary["record_type"],
                         "post_exact_gate_up_ffn_stage_trace_audit")
        self.assertEqual((len(raw), summary["process_rows"]), (8, 8))
        self.assertEqual(summary["first_nonzero_stage"],
                         "inference.cached_prefill.blocks.0.ffn.down")
        self.assertTrue(summary["all_repeat_metrics_equal"])
        for case in summary["cases"]:
            for stage in case["stages"][:4]:
                self.assertTrue(stage["b1_vs_batch_row0"]["bitwise_equal"])
                self.assertTrue(stage["batch_row0_vs_row1"]["bitwise_equal"])
            if case["batch"] > 1:
                self.assertFalse(case["stages"][4][
                    "b1_vs_batch_row0"]["bitwise_equal"])
        b2 = next(case for case in summary["cases"] if case["batch"] == 2)
        self.assertEqual(b2["stages"][4]["b1_vs_batch_row0"]["maximum"],
                         1.71661376953125e-05)
        self.assertTrue(analysis[
            "all_norm_gate_up_activation_cross_batch_bitwise_equal"])
        self.assertEqual(verification["scope_dispatches_per_process"], 224)
        self.assertFalse(list(root.glob("*.bin")))
        ET.parse(root / "post-exact-gate-up-trace.svg")

    def test_current_all_exact_ffn_gate_rejects_rms_worsening(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-fp32-prefill-ffn-all-exact-gate")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        precision = [json.loads(line) for line in
                     (root / "precision-raw.jsonl").read_text(
                         encoding="utf-8").splitlines() if line]
        performance = [json.loads(line) for line in
                       (root / "performance-raw.jsonl").read_text(
                           encoding="utf-8").splitlines() if line]
        policies = {row["policy"]: row for row in summary["policy_summaries"]}
        self.assertEqual(summary["record_type"],
                         "prefill_ffn_gate_up_all_exact_model_gate")
        self.assertEqual((len(precision), len(performance)), (16, 16))
        self.assertTrue(summary["robust_logit_max_improvement"])
        self.assertFalse(summary["robust_logit_rms_improvement"])
        self.assertTrue(summary["performance_gate_passed"])
        self.assertFalse(summary["candidate_admitted"])
        self.assertAlmostEqual(summary["candidate_minimum_prefill_speedup"],
                               0.9639047043166205)
        self.assertEqual(
            policies["selective-ffn-exact"][
                "maximum_logit_cross_batch_error"],
            0.0008723735809326172)
        self.assertEqual(
            policies["selective-ffn-exact"][
                "maximum_logit_cross_batch_rms_error"],
            0.00024268345756307228)
        self.assertAlmostEqual(analysis["maximum_improvement_fraction"],
                               0.35546943808349485)
        self.assertAlmostEqual(analysis["rms_worsening_fraction"],
                               0.0578331587000982)
        self.assertFalse(verification["admission"]["passed"])
        ET.parse(root / "ffn-all-exact-model-gate.svg")

    def test_current_prefill_ffn_model_gate_rejects_rms(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-fp32-prefill-ffn-model-gate")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        precision = [json.loads(line) for line in
                     (root / "precision-raw.jsonl").read_text(
                         encoding="utf-8").splitlines() if line]
        performance = [json.loads(line) for line in
                       (root / "performance-raw.jsonl").read_text(
                           encoding="utf-8").splitlines() if line]
        policies = {row["policy"]: row for row in summary["policy_summaries"]}
        self.assertEqual((len(precision), len(performance)), (16, 16))
        self.assertTrue(summary["robust_logit_max_improvement"])
        self.assertFalse(summary["robust_logit_rms_improvement"])
        self.assertTrue(summary["performance_gate_passed"])
        self.assertFalse(summary["candidate_admitted"])
        self.assertAlmostEqual(summary["candidate_minimum_prefill_speedup"],
                               0.9805529229183381)
        self.assertEqual(
            policies["upstream"]["maximum_logit_cross_batch_error"],
            0.0013535022735595703)
        self.assertEqual(
            policies["selective-ffn-exact"][
                "maximum_logit_cross_batch_error"],
            0.0011911392211914062)
        self.assertAlmostEqual(analysis["maximum_improvement_fraction"],
                               0.11995772415007921)
        self.assertAlmostEqual(analysis["rms_improvement_fraction"],
                               0.033310906317173705)
        for batch in summary["batches"]:
            baseline = next(row for row in summary["cases"]
                            if row["policy"] == "upstream" and
                            row["batch"] == batch)
            candidate = next(row for row in summary["cases"]
                             if row["policy"] == "selective-ffn-exact" and
                             row["batch"] == batch)
            self.assertEqual(baseline["engine_peak_bytes_maximum"],
                             candidate["engine_peak_bytes_maximum"])
            self.assertEqual(
                baseline["engine_backend_allocation_calls_maximum"],
                candidate["engine_backend_allocation_calls_maximum"])
        self.assertFalse(verification["admission"]["passed"])
        ET.parse(root / "ffn-model-gate.svg")

    def test_current_ffn_solution_matrix_rejects_m8192(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-fp32-ffn-row-invariance")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        inventory = json.loads((root / "inventory.json").read_text(
            encoding="utf-8"))
        raw = [json.loads(line) for line in
               (root / "raw.jsonl").read_text(encoding="utf-8").splitlines()
               if line]
        self.assertEqual(summary["rows"], [2048, 4096, 8192, 16384])
        self.assertEqual(summary["common_candidate_count"], 33)
        self.assertEqual(len(raw), 33)
        self.assertEqual(summary["block_invariant_indices"], [296100])
        self.assertEqual(summary["performance_admitted_count"], 0)
        self.assertEqual(summary["recommended_index"], -1)
        exact = next(row for row in summary["candidates"]
                     if row["index"] == 296100)
        self.assertTrue(exact["block_invariant"])
        self.assertEqual(exact["block_maximum_error"], 0.0)
        self.assertEqual(exact["speedup_vs_default"], [
            1.039587677, 0.950527234, 0.941296214, 0.995181022])
        self.assertAlmostEqual(exact["minimum_speedup"], 0.941296214)
        self.assertEqual(inventory["default_event_ms_p50"],
                         summary["default_event_ms_p50"])
        self.assertEqual(analysis["block_invariant_index"], 296100)
        self.assertTrue(verification["correctness_before_timing"])
        ET.parse(root / "ffn-row-invariance.svg")

    def test_ffn_solution_summary_requires_exact_and_every_m_performance(self):
        candidates = []
        for index, exact, speeds in (
                (10, True, [1.01, 1.02, 1.03, 1.04]),
                (11, True, [1.20, 1.10, 0.94, 1.30]),
                (12, False, [2.0, 2.0, 2.0, 2.0])):
            candidates.append({
                "index": index, "maximum_workspace_bytes": 0,
                "supported": True, "sentinel_passed": True,
                "block_invariant": exact,
                "sentinel_maximum_error": 0.0,
                "sentinel_rms_error": 0.0,
                "block_maximum_error": 0.0 if exact else 0.01,
                "block_rms_error": 0.0 if exact else 0.001,
                "event_ms_p50": [10.0 / value for value in speeds],
                "speedup_vs_default": speeds,
            })
        inventory = {
            "default_event_ms_p50": [10.0] * 4,
            "common_candidate_count": 3, "supported_count": 3,
            "sentinel_pass_count": 3, "block_invariant_count": 2,
            "candidates": candidates,
        }
        summary = FFN_SOLUTIONS.summarize(inventory)
        self.assertEqual(summary["block_invariant_indices"], [10, 11])
        self.assertEqual(summary["performance_admitted_indices"], [10])
        self.assertEqual(summary["recommended_index"], 10)
        self.assertAlmostEqual(summary["recommended_minimum_speedup"], 1.01)
        ET.fromstring(FFN_SOLUTIONS.render(summary))

    def test_current_prefill_ffn_trace_selects_gate_and_up(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-prefill-ffn-stage-trace")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        raw = [json.loads(line) for line in
               (root / "raw.jsonl").read_text(encoding="utf-8").splitlines()
               if line]
        self.assertEqual((len(raw), summary["process_rows"]), (8, 8))
        self.assertEqual(summary["stage_count"], 7)
        self.assertEqual(summary["binary_files_retained"], 0)
        self.assertEqual(summary["first_nonzero_stage"],
                         FFN_STAGES.PREFIX + ".ffn.gate")
        self.assertTrue(summary["all_repeat_metrics_equal"])
        for case in summary["cases"]:
            norm = case["stages"][0]
            self.assertTrue(norm["b1_vs_batch_row0"]["bitwise_equal"])
            self.assertTrue(norm["batch_row0_vs_row1"]["bitwise_equal"])
            if case["batch"] > 1:
                self.assertEqual(case["first_nonzero_stage"],
                                 FFN_STAGES.PREFIX + ".ffn.gate")
                gate, up = case["stages"][1:3]
                self.assertFalse(gate["b1_vs_batch_row0"]["bitwise_equal"])
                self.assertFalse(up["b1_vs_batch_row0"]["bitwise_equal"])
        b2 = next(case for case in summary["cases"] if case["batch"] == 2)
        self.assertEqual(b2["stages"][1]["b1_vs_batch_row0"]["maximum"],
                         9.5367431640625e-06)
        self.assertTrue(analysis["up_is_independent_nonzero_projection"])
        self.assertEqual(verification["focused_current_revision"][
            "release_mi300x_processes"], "8/8")
        self.assertFalse(list(root.glob("*.bin")))
        ET.parse(root / "ffn-stage-trace.svg")

    def test_prefill_ffn_stage_runner_is_block_zero_and_binary_filtered(self):
        args = type("Args", (), {
            "binary": Path("micro"), "context": 2048,
        })()
        model = {
            "config": "config.json", "weights": "model.bin",
            "inference": {"token_ids": [1, 2]},
        }
        command = FFN_STAGES.command(
            args, model, 4, Path("trace.jsonl"), Path("cache.bin"),
            Path("values"))
        self.assertNotIn("--trace-all-layer-details", command)
        self.assertEqual(command[
            command.index("--trace-value-filter") + 1],
            ",".join(FFN_STAGES.STAGES))
        self.assertEqual(command[
            command.index("--trace-max-elements") + 1], "1")
        self.assertEqual(command[
            command.index("--trace-binary-directory") + 1], "values")
        route = {
            "status": "pass", "batch": 4, "token_count": 2048,
            "trace_record_count": 55,
            "trace_binary_record_count": 7,
            "fp32_prefill_q_solution_index": 296100,
            "fp32_prefill_kv_solution_index": 292135,
            "fp32_prefill_attention_qk_solution_index": 304681,
            "fp32_prefill_attention_pv_solution_index": 295716,
            "fp32_prefill_attention_o_solution_index": 296100,
            "fp32_solution_registered_entries": 5,
            "fp32_solution_cached_algorithms": 5,
            "fp32_solution_registry_hits": 168,
            "fp32_solution_cache_misses": 5,
            "fp32_solution_cache_hits": 163,
            "fp32_solution_dispatches": 168,
        }
        FFN_STAGES.require_route(route, 4)

    def test_current_exact_stack_gate_closes_composition_track(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-fp32-prefill-exact-stack-gate")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        precision = [json.loads(line) for line in
                     (root / "precision-raw.jsonl").read_text(
                         encoding="utf-8").splitlines() if line]
        performance = [json.loads(line) for line in
                       (root / "performance-raw.jsonl").read_text(
                           encoding="utf-8").splitlines() if line]
        policies = {row["policy"]: row for row in summary["policy_summaries"]}
        self.assertEqual((len(precision), len(performance)), (16, 16))
        self.assertFalse(summary["robust_logit_max_improvement"])
        self.assertFalse(summary["robust_logit_rms_improvement"])
        self.assertTrue(summary["performance_gate_passed"])
        self.assertFalse(summary["candidate_admitted"])
        self.assertAlmostEqual(summary["candidate_minimum_prefill_speedup"],
                               0.9867002251579743)
        self.assertEqual(
            policies["upstream-exact"]["maximum_logit_cross_batch_error"],
            0.0012532472610473633)
        self.assertEqual(
            policies["batch-selective"]["maximum_logit_cross_batch_error"],
            0.0013401508331298828)
        self.assertAlmostEqual(analysis["maximum_change_fraction"],
                               0.06934271853895169)
        self.assertAlmostEqual(analysis["rms_improvement_fraction"],
                               0.02492031334239908)
        for batch in summary["batches"]:
            baseline = next(row for row in summary["cases"]
                            if row["policy"] == "upstream-exact" and
                            row["batch"] == batch)
            candidate = next(row for row in summary["cases"]
                             if row["policy"] == "batch-selective" and
                             row["batch"] == batch)
            self.assertEqual(baseline["engine_peak_bytes_maximum"],
                             candidate["engine_peak_bytes_maximum"])
            self.assertEqual(
                baseline["engine_backend_allocation_calls_maximum"],
                candidate["engine_backend_allocation_calls_maximum"])
        self.assertEqual(verification["build_type"], "Release")
        self.assertFalse(verification["admission"]["passed"])
        ET.parse(root / "exact-stack-gate.svg")

    def test_exact_stack_routes_are_fixed_before_measurement(self):
        args = type("Args", (), {
            "binary": Path("micro"), "context": 2048,
        })()
        model = {
            "config": "config.json", "weights": "model.bin",
            "inference": {"token_ids": [1, 2]},
        }
        b1 = EXACT_STACK.command(
            args, model, "batch-selective", 1, 0)
        self.assertNotIn(
            "--fp32-prefill-attention-qk-solution-index", b1)
        self.assertNotIn(
            "--fp32-prefill-attention-o-solution-index", b1)
        b2 = EXACT_STACK.command(
            args, model, "batch-selective", 2, 0)
        self.assertEqual(b2[
            b2.index("--fp32-prefill-attention-qk-solution-index") + 1],
            "304681")
        self.assertEqual(b2[
            b2.index("--fp32-prefill-attention-pv-solution-index") + 1],
            "295716")
        self.assertEqual(b2[
            b2.index("--fp32-prefill-attention-o-solution-index") + 1],
            "296100")
        b8 = EXACT_STACK.command(
            args, model, "batch-selective", 8, 0)
        self.assertNotIn(
            "--fp32-prefill-attention-o-solution-index", b8)

        route = {
            "status": "pass", "batch": 2, "token_count": 2048,
            "decode_tokens": 1, "kv_cache_dtype": "bf16",
            "fp32_prefill_q_solution_index": 296100,
            "fp32_prefill_kv_solution_index": 292135,
            "fp32_prefill_attention_qk_solution_index": 304681,
            "fp32_prefill_attention_pv_solution_index": 295716,
            "fp32_prefill_attention_o_solution_index": 296100,
            "fp32_solution_registered_entries": 5,
            "fp32_solution_cached_algorithms": 5,
            "fp32_solution_registry_hits": 168,
            "fp32_solution_cache_misses": 5,
            "fp32_solution_cache_hits": 163,
            "fp32_solution_dispatches": 168,
        }
        EXACT_STACK.require_route(
            route, "batch-selective", 2, 2048, 0)

    def test_current_o_model_gate_rejects_b1_performance(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-fp32-prefill-o-model-gate")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        precision = [json.loads(line) for line in
                     (root / "precision-raw.jsonl").read_text(
                         encoding="utf-8").splitlines() if line]
        performance = [json.loads(line) for line in
                       (root / "performance-raw.jsonl").read_text(
                           encoding="utf-8").splitlines() if line]
        policies = {row["policy"]: row for row in summary["policy_summaries"]}
        self.assertEqual((len(precision), len(performance)), (16, 16))
        self.assertTrue(summary["robust_logit_max_improvement"])
        self.assertTrue(summary["robust_logit_rms_improvement"])
        self.assertFalse(summary["performance_gate_passed"])
        self.assertFalse(summary["candidate_admitted"])
        self.assertAlmostEqual(summary["candidate_minimum_prefill_speedup"],
                               0.9439958736379434)
        self.assertEqual(
            policies["exact-core"]["maximum_logit_cross_batch_error"],
            0.0015616416931152344)
        self.assertEqual(
            policies["exact-core-o"]["maximum_logit_cross_batch_error"],
            0.0011752843856811523)
        self.assertAlmostEqual(analysis["maximum_improvement_fraction"],
                               0.24739944153562727)
        self.assertAlmostEqual(analysis["rms_improvement_fraction"],
                               0.3256643517300204)
        for batch in summary["batches"]:
            baseline = next(row for row in summary["cases"]
                            if row["policy"] == "exact-core" and
                            row["batch"] == batch)
            candidate = next(row for row in summary["cases"]
                             if row["policy"] == "exact-core-o" and
                             row["batch"] == batch)
            self.assertEqual(baseline["engine_peak_bytes_maximum"],
                             candidate["engine_peak_bytes_maximum"])
            self.assertEqual(
                baseline["engine_backend_allocation_calls_maximum"],
                candidate["engine_backend_allocation_calls_maximum"])
        self.assertFalse(verification["admission"]["passed"])
        ET.parse(root / "o-model-gate.svg")

    def test_o_model_gate_command_and_registry_counts(self):
        args = type("Args", (), {
            "binary": Path("micro"), "context": 2048,
        })()
        model = {
            "config": "config.json", "weights": "model.bin",
            "inference": {"token_ids": [1, 2]},
        }
        baseline = O_MODEL.command(
            args, model, "upstream-exact", 2, 0)
        self.assertNotIn("--fp32-prefill-attention-o-solution-index", baseline)
        candidate = O_MODEL.command(
            args, model, "attention-exact", 2, 0)
        self.assertEqual(candidate[
            candidate.index("--fp32-prefill-attention-o-solution-index") + 1],
            "296100")
        route = {
            "status": "pass", "batch": 2, "token_count": 2048,
            "decode_tokens": 1, "kv_cache_dtype": "bf16",
            "fp32_prefill_q_solution_index": 296100,
            "fp32_prefill_kv_solution_index": 292135,
            "fp32_prefill_attention_qk_solution_index": 304681,
            "fp32_prefill_attention_pv_solution_index": 295716,
            "fp32_prefill_attention_o_solution_index": 296100,
            "fp32_solution_registered_entries": 5,
            "fp32_solution_cached_algorithms": 5,
            "fp32_solution_registry_hits": 336,
            "fp32_solution_cache_misses": 5,
            "fp32_solution_cache_hits": 331,
            "fp32_solution_dispatches": 336,
        }
        O_MODEL.require_route(route, "attention-exact", 2, 2048, 1)

    def test_current_post_exact_o_trace_locates_ffn_output(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-post-exact-o-block0-trace")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        self.assertEqual(summary["process_rows"], 8)
        self.assertEqual(summary["first_nonzero_after_cache"],
                         POST_EXACT_O.TRACE.PREFIX + ".ffn_output")
        for case in summary["cases"]:
            for suffix in ("attention.context", "attention.output",
                           "attention_residual", "ffn_norm"):
                stage = next(row for row in case["stages"]
                             if row["name"].endswith(suffix))
                self.assertTrue(stage["b1_vs_batch_row0"]["bitwise_equal"])
                self.assertTrue(stage["batch_row0_vs_row1"]["bitwise_equal"])
        b2 = next(case for case in summary["cases"] if case["batch"] == 2)
        ffn = next(row for row in b2["stages"]
                   if row["name"].endswith(".ffn_output"))
        self.assertEqual(ffn["b1_vs_batch_row0"]["maximum"],
                         2.193450927734375e-05)
        self.assertTrue(analysis["o_scope_causal_effect_supported"])
        self.assertEqual(verification["runner_commit"],
                         "89b9bb8ea14b36089ce43bd44dfc91dfcfc86f04")
        ET.parse(root / "post-exact-o-trace.svg")

    def test_post_exact_o_trace_command_and_route_are_scoped(self):
        args = type("Args", (), {
            "binary": Path("micro"), "context": 2048,
        })()
        model = {
            "config": "config.json", "weights": "model.bin",
            "inference": {"token_ids": [1, 2]},
        }
        command = POST_EXACT_O.command(
            args, model, 4, Path("trace.jsonl"), Path("cache.bin"))
        self.assertEqual(command[
            command.index("--fp32-prefill-attention-o-solution-index") + 1],
            "296100")
        route = {
            "status": "pass", "batch": 4, "token_count": 2048,
            "trace_record_count": 50,
            "fp32_prefill_q_solution_index": 296100,
            "fp32_prefill_kv_solution_index": 292135,
            "fp32_prefill_attention_qk_solution_index": 304681,
            "fp32_prefill_attention_pv_solution_index": 295716,
            "fp32_prefill_attention_o_solution_index": 296100,
            "fp32_solution_registered_entries": 5,
            "fp32_solution_cached_algorithms": 5,
            "fp32_solution_registry_hits": 168,
            "fp32_solution_cache_misses": 5,
            "fp32_solution_cache_hits": 163,
            "fp32_solution_dispatches": 168,
        }
        POST_EXACT_O.require_route(route, 4)

    def test_current_post_exact_core_trace_locates_o_projection(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-post-exact-core-block0-trace")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        self.assertEqual(summary["process_rows"], 8)
        self.assertEqual(summary["stage_count"], 17)
        self.assertEqual(summary["first_nonzero_after_cache"],
                         POST_EXACT_CORE.BASE.PREFIX + ".attention.output")
        for case in summary["cases"]:
            context = next(stage for stage in case["stages"]
                           if stage["name"].endswith(".attention.context"))
            self.assertTrue(context["b1_vs_batch_row0"]["bitwise_equal"])
            self.assertTrue(context["batch_row0_vs_row1"]["bitwise_equal"])
        b2 = next(case for case in summary["cases"] if case["batch"] == 2)
        output = next(stage for stage in b2["stages"]
                      if stage["name"].endswith(".attention.output"))
        self.assertEqual(output["b1_vs_batch_row0"]["maximum"],
                         3.337860107421875e-05)
        self.assertFalse(output["batch_row0_vs_row1"]["bitwise_equal"])
        self.assertTrue(analysis["all_context_cross_batch_bitwise_equal"])
        self.assertTrue(analysis[
            "within_batch_first_divergence_is_attention_output"])
        self.assertEqual(verification["runner_commit"],
                         "836543746efed32bcc72b231ec195bef456fe3ea")
        ET.parse(root / "post-exact-core-trace.svg")

    def test_post_exact_core_trace_command_and_route_are_scoped(self):
        args = type("Args", (), {
            "binary": Path("micro"), "context": 2048,
        })()
        model = {
            "config": "config.json", "weights": "model.bin",
            "inference": {"token_ids": [1, 2]},
        }
        command = POST_EXACT_CORE.command(
            args, model, 4, Path("trace.jsonl"), Path("cache.bin"))
        self.assertEqual(command[
            command.index("--fp32-prefill-attention-qk-solution-index") + 1],
            "304681")
        self.assertEqual(command[
            command.index("--fp32-prefill-attention-pv-solution-index") + 1],
            "295716")
        route = {
            "status": "pass", "batch": 4, "token_count": 2048,
            "trace_record_count": 50,
            "fp32_prefill_q_solution_index": 296100,
            "fp32_prefill_kv_solution_index": 292135,
            "fp32_prefill_attention_qk_solution_index": 304681,
            "fp32_prefill_attention_pv_solution_index": 295716,
            "fp32_solution_registered_entries": 4,
            "fp32_solution_cached_algorithms": 4,
            "fp32_solution_registry_hits": 140,
            "fp32_solution_cache_misses": 4,
            "fp32_solution_cache_hits": 136,
            "fp32_solution_dispatches": 140,
        }
        POST_EXACT_CORE.require_route(route, 4)

    def test_current_selective_attention_gate_closes_solution_track(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-fp32-prefill-attention-selective-gate")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        precision = [json.loads(line) for line in
                     (root / "precision-raw.jsonl").read_text(
                         encoding="utf-8").splitlines() if line]
        performance = [json.loads(line) for line in
                       (root / "performance-raw.jsonl").read_text(
                           encoding="utf-8").splitlines() if line]
        policies = {row["policy"]: row for row in summary["policy_summaries"]}
        self.assertEqual((len(precision), len(performance)), (16, 16))
        self.assertTrue(summary["performance_gate_passed"])
        self.assertFalse(summary["robust_logit_max_improvement"])
        self.assertTrue(summary["robust_logit_rms_improvement"])
        self.assertFalse(summary["candidate_admitted"])
        self.assertAlmostEqual(summary["candidate_minimum_prefill_speedup"],
                               0.9941303826374912)
        self.assertEqual(
            policies["upstream-exact"]["maximum_logit_cross_batch_error"],
            0.0012532472610473633)
        self.assertEqual(
            policies["batch-selective"]["maximum_logit_cross_batch_error"],
            0.0011773109436035156)
        self.assertEqual(
            policies["batch-selective"]["maximum_logit_cross_batch_rms_error"],
            0.0002281133416539806)
        b2_base = next(row for row in summary["cases"]
                       if row["policy"] == "upstream-exact" and row["batch"] == 2)
        b2_candidate = next(row for row in summary["cases"]
                            if row["policy"] == "batch-selective" and row["batch"] == 2)
        self.assertGreater(b2_candidate["logits_cross_batch"]["maximum"],
                           b2_base["logits_cross_batch"]["maximum"])
        for batch in summary["batches"]:
            base = next(row for row in summary["cases"]
                        if row["policy"] == "upstream-exact" and
                        row["batch"] == batch)
            candidate = next(row for row in summary["cases"]
                             if row["policy"] == "batch-selective" and
                             row["batch"] == batch)
            self.assertEqual(base["engine_peak_bytes_maximum"],
                             candidate["engine_peak_bytes_maximum"])
            self.assertEqual(base["engine_backend_allocation_calls_maximum"],
                             candidate["engine_backend_allocation_calls_maximum"])
        self.assertTrue(analysis["attention_solution_track_closed"])
        self.assertEqual(verification["engine_commit"],
                         "6a31cfeed33bf76456e6ad81f4d3b87fbd8076a7")
        self.assertEqual(verification["runner_commit"],
                         "61d9e839aec2d85eaa735befe59d7cc23dc26f7b")
        ET.parse(root / "selective-gate.svg")

    def test_selective_attention_gate_checks_exact_batch_routes(self):
        args = type("Args", (), {
            "binary": Path("micro"), "context": 2048,
        })()
        model = {
            "config": "config.json", "weights": "model.bin",
            "inference": {"token_ids": [1, 2]},
        }
        b1 = ATTENTION_SELECTIVE.command(
            args, model, "batch-selective", 1, 0)
        self.assertNotIn(
            "--fp32-prefill-attention-qk-solution-index", b1)
        b4 = ATTENTION_SELECTIVE.command(
            args, model, "batch-selective", 4, 0)
        self.assertEqual(b4[
            b4.index("--fp32-prefill-attention-qk-solution-index") + 1],
            "311274")
        self.assertEqual(b4[
            b4.index("--fp32-prefill-attention-pv-solution-index") + 1],
            "295716")
        route = {
            "status": "pass", "batch": 4, "token_count": 2048,
            "decode_tokens": 1, "kv_cache_dtype": "bf16",
            "fp32_prefill_q_solution_index": 296100,
            "fp32_prefill_kv_solution_index": 292135,
            "fp32_prefill_attention_qk_solution_index": 311274,
            "fp32_prefill_attention_pv_solution_index": 295716,
            "fp32_solution_registered_entries": 4,
            "fp32_solution_cached_algorithms": 4,
            "fp32_solution_registry_hits": 280,
            "fp32_solution_cache_misses": 4,
            "fp32_solution_cache_hits": 276,
            "fp32_solution_dispatches": 280,
        }
        ATTENTION_SELECTIVE.require_route(
            route, "batch-selective", 4, 2048, 1)

        exact = {"bitwise_equal": True, "maximum": 0.0, "rms": 0.0}
        precision = []
        performance = []
        for policy in ATTENTION_SELECTIVE.POLICIES:
            for batch in ATTENTION_SELECTIVE.BATCHES:
                indices = (ATTENTION_SELECTIVE.SELECTIVE[batch]
                           if policy == "batch-selective" else {"qk": -1, "pv": -1})
                for run in (1, 2):
                    changed = batch > 1
                    logits = {
                        "bitwise_equal": not changed,
                        "maximum": (0.005 if policy == "batch-selective" else 0.01)
                                   if changed else 0.0,
                        "rms": (0.002 if policy == "batch-selective" else 0.005)
                               if changed else 0.0,
                    }
                    precision.append({
                        "policy": policy, "batch": batch,
                        "process_run": run,
                        "qk_solution_index": indices["qk"],
                        "pv_solution_index": indices["pv"],
                        "key_cross_batch": exact,
                        "value_cross_batch": exact,
                        "logits_cross_batch": logits,
                        "key_vs_upstream": exact,
                        "value_vs_upstream": exact,
                        "logits_vs_upstream": exact,
                        "key_within_batch_bitwise_equal": True,
                        "value_within_batch_bitwise_equal": True,
                        "logits_within_batch_bitwise_equal": True,
                    })
                    base_ms = 100.0 * batch
                    prefill_ms = (base_ms if policy == "upstream-exact"
                                  else base_ms / 1.02)
                    performance.append({
                        "policy": policy, "batch": batch,
                        "process_run": run,
                        "prefill_ms": prefill_ms,
                        "prefill_tokens_per_second":
                            batch * 2048 * 1000.0 / prefill_ms,
                        "engine_peak_bytes": 1000,
                        "engine_backend_allocation_calls": 10,
                        "generated_tokens": [1],
                    })
        summary = ATTENTION_SELECTIVE.summarize(precision, performance)
        self.assertTrue(summary["robust_logit_max_improvement"])
        self.assertTrue(summary["robust_logit_rms_improvement"])
        self.assertTrue(summary["performance_gate_passed"])
        self.assertTrue(summary["candidate_admitted"])
        ET.fromstring(ATTENTION_SELECTIVE.render(summary))

    def test_current_prefill_attention_model_gate_rejects_exact_pair(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-fp32-prefill-attention-model-gate")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        precision = [json.loads(line) for line in
                     (root / "precision-raw.jsonl").read_text(
                         encoding="utf-8").splitlines() if line]
        performance = [json.loads(line) for line in
                       (root / "performance-raw.jsonl").read_text(
                           encoding="utf-8").splitlines() if line]
        policies = {row["policy"]: row for row in summary["policy_summaries"]}
        self.assertEqual((len(precision), len(performance)), (16, 16))
        self.assertTrue(summary["candidate_core_bitwise_equal"])
        self.assertFalse(summary["candidate_admitted"])
        self.assertFalse(summary["robust_logit_max_improvement"])
        self.assertFalse(summary["robust_logit_rms_improvement"])
        self.assertFalse(summary["performance_gate_passed"])
        self.assertEqual(
            policies["upstream-exact"]["maximum_logit_cross_batch_error"],
            0.0012532472610473633)
        self.assertEqual(
            policies["attention-exact"]["maximum_logit_cross_batch_error"],
            0.0015616416931152344)
        self.assertAlmostEqual(summary["candidate_minimum_prefill_speedup"],
                               0.9495440415562535)
        for batch in summary["batches"]:
            base = next(row for row in summary["cases"]
                        if row["policy"] == "upstream-exact" and
                        row["batch"] == batch)
            candidate = next(row for row in summary["cases"]
                             if row["policy"] == "attention-exact" and
                             row["batch"] == batch)
            self.assertEqual(base["engine_peak_bytes_maximum"],
                             candidate["engine_peak_bytes_maximum"])
            self.assertEqual(base["engine_backend_allocation_calls_maximum"],
                             candidate["engine_backend_allocation_calls_maximum"])
        self.assertFalse(analysis["candidate_admitted"])
        self.assertEqual(verification["engine_commit"],
                         "6a31cfeed33bf76456e6ad81f4d3b87fbd8076a7")
        self.assertEqual(verification["runner_commit"],
                         "22aa57af5fa656cc0e601a72279b5ef389a59f8f")
        self.assertFalse(any(root.glob("*.bin")))
        ET.parse(root / "model-gate.svg")

    def test_attention_model_gate_requires_core_logits_and_performance(self):
        args = type("Args", (), {
            "binary": Path("micro"), "context": 2048,
        })()
        model = {
            "config": "config.json", "weights": "model.bin",
            "inference": {"token_ids": [1, 2]},
        }
        candidate = ATTENTION_MODEL.command(
            args, model, "attention-exact", 2, 0)
        self.assertEqual(candidate[
            candidate.index("--fp32-prefill-attention-qk-solution-index") + 1],
            "304681")
        self.assertEqual(candidate[
            candidate.index("--fp32-prefill-attention-pv-solution-index") + 1],
            "295716")
        route = {
            "status": "pass", "batch": 2, "token_count": 2048,
            "decode_tokens": 1, "kv_cache_dtype": "bf16",
            "fp32_prefill_q_solution_index": 296100,
            "fp32_prefill_kv_solution_index": 292135,
            "fp32_prefill_attention_qk_solution_index": 304681,
            "fp32_prefill_attention_pv_solution_index": 295716,
            "fp32_solution_registered_entries": 4,
            "fp32_solution_cached_algorithms": 4,
            "fp32_solution_registry_hits": 280,
            "fp32_solution_cache_misses": 4,
            "fp32_solution_cache_hits": 276,
            "fp32_solution_dispatches": 280,
        }
        ATTENTION_MODEL.require_route(
            route, "attention-exact", 2, 2048, 1)

        precision = []
        performance = []
        exact_metric = {"bitwise_equal": True, "maximum": 0.0, "rms": 0.0}
        for policy in ATTENTION_MODEL.POLICIES:
            for batch in ATTENTION_MODEL.BATCHES:
                for run in (1, 2):
                    changed = policy == "upstream-exact" and batch > 1
                    logit_metric = {
                        "bitwise_equal": not changed,
                        "maximum": 0.01 if changed else 0.001 if batch > 1 else 0.0,
                        "rms": 0.005 if changed else 0.0005 if batch > 1 else 0.0,
                    }
                    core = []
                    if policy == "attention-exact":
                        core = [{
                            "name": name,
                            "b1_vs_batch_row0": exact_metric,
                            "batch_row0_vs_row1": exact_metric,
                        } for name in ATTENTION_MODEL.CORE.STAGES]
                    precision.append({
                        "policy": policy, "batch": batch,
                        "process_run": run,
                        "key_cross_batch": exact_metric,
                        "value_cross_batch": exact_metric,
                        "logits_cross_batch": logit_metric,
                        "key_vs_upstream": exact_metric,
                        "value_vs_upstream": exact_metric,
                        "logits_vs_upstream": exact_metric,
                        "key_within_batch_bitwise_equal": True,
                        "value_within_batch_bitwise_equal": True,
                        "logits_within_batch_bitwise_equal": True,
                        "core_stages": core,
                    })
                    base_ms = 100.0 * batch
                    prefill_ms = (base_ms if policy == "upstream-exact"
                                  else base_ms * 0.96)
                    performance.append({
                        "policy": policy, "batch": batch,
                        "process_run": run,
                        "prefill_ms": prefill_ms,
                        "prefill_tokens_per_second":
                            batch * 2048 * 1000.0 / prefill_ms,
                        "engine_peak_bytes": 1000,
                        "engine_backend_allocation_calls": 10,
                        "generated_tokens": [1],
                    })
        summary = ATTENTION_MODEL.summarize(precision, performance)
        self.assertTrue(summary["candidate_core_bitwise_equal"])
        self.assertTrue(summary["robust_logit_max_improvement"])
        self.assertTrue(summary["robust_logit_rms_improvement"])
        self.assertTrue(summary["performance_gate_passed"])
        self.assertTrue(summary["candidate_admitted"])
        ET.fromstring(ATTENTION_MODEL.render(summary))

    def test_current_attention_solution_matrix_rejects_all_defaults(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-fp32-attention-batch-invariance")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        raw = [json.loads(line) for line in (root / "raw.jsonl").read_text(
            encoding="utf-8").splitlines() if line]
        operations = {row["operation"]: row for row in summary["operations"]}
        self.assertEqual(summary["backend_batch_counts"], [12, 24, 48, 96])
        self.assertEqual(len(raw), 36)
        self.assertEqual(operations["qk"]["common_candidate_count"], 34)
        self.assertEqual(operations["qk"]["block_invariant_count"], 34)
        self.assertEqual(operations["qk"]["best_exact_index"], 304681)
        self.assertEqual(operations["qk"]["admitted_index"], -1)
        self.assertAlmostEqual(
            operations["qk"]["best_exact_minimum_event_speedup"],
            0.915873395648)
        self.assertEqual(operations["pv"]["common_candidate_count"], 2)
        self.assertEqual(operations["pv"]["block_invariant_count"], 2)
        self.assertEqual(operations["pv"]["best_exact_index"], 295716)
        self.assertEqual(operations["pv"]["admitted_index"], -1)
        self.assertAlmostEqual(
            operations["pv"]["best_exact_minimum_event_speedup"],
            0.535391731132)
        self.assertFalse(operations["qk"]["default_block_invariant"])
        self.assertFalse(operations["pv"]["default_block_invariant"])
        self.assertFalse(analysis["default_change_admitted"])
        self.assertEqual(verification["engine_commit"],
                         "8317d2b455bd750b37fc90f78aaaa2d2d2436be8")
        self.assertEqual(verification["qk_admitted_index"], -1)
        self.assertEqual(verification["pv_admitted_index"], -1)
        ET.parse(root / "attention-solutions.svg")

    def test_attention_solution_summary_prefers_exact_non_regressing_candidate(self):
        inventories = {}
        results = {}
        for operation in ("qk", "pv"):
            inventories[operation] = {
                "common_candidate_indices": [7, 9],
            }
            candidates = []
            exact_speedups = ([1.1, 1.0, 1.2, 1.05] if operation == "qk"
                              else [0.5, 1.0, 1.2, 1.05])
            for index, exact, speedups in (
                    (7, True, exact_speedups),
                    (9, False, [])):
                candidates.append({
                    "index": index,
                    "maximum_workspace_bytes": 0,
                    "block_invariant": exact,
                    "event_speedup_vs_default": speedups,
                })
            results[operation] = {
                "candidates": candidates,
                "shape_candidate_counts": [64, 64, 64, 64],
                "correctness_passed_count": 2,
                "block_invariant_count": 1,
                "default_block_invariant": False,
                "default_block_maximum_error": 0.1,
                "default_block_rms_error": 0.01,
            }
        summary = ATTENTION_SOLUTIONS.summarize(results, inventories)
        self.assertEqual([row["best_exact_index"]
                          for row in summary["operations"]], [7, 7])
        self.assertEqual([row["admitted_index"]
                          for row in summary["operations"]], [7, -1])
        self.assertEqual([row["non_regressing_invariant_count"]
                          for row in summary["operations"]], [1, 0])
        ET.fromstring(ATTENTION_SOLUTIONS.render(summary))

    def test_current_prefill_attention_core_has_two_batch_boundaries(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-prefill-attention-core-matrix")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        raw = [json.loads(line) for line in (root / "raw.jsonl").read_text(
            encoding="utf-8").splitlines() if line]
        self.assertEqual(summary["process_rows"], 8)
        self.assertEqual(summary["stage_count"], 3)
        self.assertEqual(summary["binary_files_retained"], 0)
        self.assertTrue(summary["all_repeat_metrics_equal"])
        self.assertEqual(summary["first_causal_nonzero_stage"],
                         ATTENTION_CORE.STAGES[0])
        self.assertEqual(summary["first_causal_nonzero_stage_by_batch"], {
            "1": None,
            "2": ATTENTION_CORE.STAGES[2],
            "4": ATTENTION_CORE.STAGES[0],
            "8": ATTENTION_CORE.STAGES[0],
        })
        b2 = next(case for case in summary["cases"] if case["batch"] == 2)
        self.assertTrue(b2["stages"][0]
                        ["b1_vs_batch_row0_causal_visible"]["bitwise_equal"])
        self.assertTrue(b2["stages"][1]
                        ["b1_vs_batch_row0_causal_visible"]["bitwise_equal"])
        self.assertEqual(b2["stages"][2]["b1_vs_batch_row0"]["maximum"],
                         9.775161743164062e-06)
        b4 = next(case for case in summary["cases"] if case["batch"] == 4)
        self.assertEqual(b4["stages"][0]
                         ["b1_vs_batch_row0_causal_visible"]
                         ["first_numeric_index"], 2048)
        self.assertEqual(b4["stages"][0]
                         ["b1_vs_batch_row0_causal_visible"]["maximum"],
                         0.03125)
        self.assertTrue(all(case["all_within_batch_bitwise_equal"]
                            for case in summary["cases"]))
        self.assertEqual(len(raw), 8)
        self.assertTrue(analysis["masked_only_explanation_rejected"])
        self.assertTrue(analysis[
            "single_pv_only_fix_sufficient_for_all_batches_rejected"])
        self.assertEqual(verification["engine_commit"],
                         "34e78638b743c1aae9eefe7653ac75a9189e6f8c")
        self.assertFalse(any(root.glob("*.bin")))
        ET.parse(root / "attention-core.svg")

    def test_prefill_attention_core_binary_comparison_and_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left.bin"
            exact = root / "exact.bin"
            changed = root / "changed.bin"
            left.write_bytes(struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))
            exact.write_bytes(struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))
            changed.write_bytes(struct.pack("<4f", 1.0, 2.0, 3.5, 4.0))
            masked_only = root / "masked-only.bin"
            masked_only.write_bytes(struct.pack("<4f", 1.0, 2.5, 3.0, 4.0))
            self.assertTrue(ATTENTION_CORE.difference_binary(
                left, exact, 4)["bitwise_equal"])
            difference = ATTENTION_CORE.difference_binary(left, changed, 4)
            self.assertFalse(difference["bitwise_equal"])
            self.assertEqual(difference["first_bitwise_index"], 2)
            self.assertEqual(difference["first_numeric_index"], 2)
            self.assertEqual(difference["maximum"], 0.5)
            causal = ATTENTION_CORE.difference_binary(
                left, masked_only, 4, causal_sequence=2)
            self.assertTrue(causal["bitwise_equal"])
            self.assertEqual(causal["elements"], 3)

        processes = []
        for run in (1, 2):
            for batch in ATTENTION_CORE.BATCHES:
                stages = []
                for index, name in enumerate(ATTENTION_CORE.STAGES):
                    differs = batch > 1 and index == 2
                    metric = {
                        "elements": 4,
                        "bitwise_equal": not differs,
                        "first_bitwise_index": 2 if differs else None,
                        "first_numeric_index": 2 if differs else None,
                        "maximum": 0.5 if differs else 0.0,
                        "rms": 0.25 if differs else 0.0,
                        "relative_l2": 0.1 if differs else 0.0,
                    }
                    stage = {
                        "name": name,
                        "b1_vs_batch_row0": metric,
                        "batch_row0_vs_row1": {
                            **metric, "bitwise_equal": True,
                            "first_bitwise_index": None,
                            "first_numeric_index": None,
                            "maximum": 0.0, "rms": 0.0,
                            "relative_l2": 0.0,
                        },
                    }
                    if index < 2:
                        stage["b1_vs_batch_row0_causal_visible"] = metric
                        stage["batch_row0_vs_row1_causal_visible"] = {
                            **metric, "bitwise_equal": True,
                            "first_bitwise_index": None,
                            "first_numeric_index": None,
                            "maximum": 0.0, "rms": 0.0,
                            "relative_l2": 0.0,
                        }
                    stages.append(stage)
                processes.append({"batch": batch, "process_run": run,
                                  "stages": stages})
        summary = ATTENTION_CORE.summarize(processes)
        self.assertTrue(summary["all_scores_bitwise_equal"])
        self.assertTrue(summary["all_probabilities_bitwise_equal"])
        self.assertEqual(summary["first_nonzero_stage"],
                         ATTENTION_CORE.STAGES[2])
        self.assertEqual(summary["first_causal_nonzero_stage"],
                         ATTENTION_CORE.STAGES[2])
        ET.fromstring(ATTENTION_CORE.render(summary))

    def test_current_post_cache_trace_locates_attention_context(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-post-cache-block0-trace")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        self.assertEqual(summary["process_rows"], 8)
        self.assertEqual(summary["stage_count"], 17)
        self.assertTrue(summary["all_cache_cross_batch_bitwise_equal"])
        self.assertEqual(summary["first_nonzero_after_cache"],
                         POST_CACHE.PREFIX + ".attention.context")
        self.assertTrue(analysis[
            "attention_context_is_first_post_cache_source_supported"])
        self.assertTrue(analysis["all_context_within_batch_rows_bitwise_equal"])
        self.assertEqual(verification["measurement_commit"],
                         "bbcaf95e00e995ad7f6313683b4e820f6012fc6e")
        ET.parse(root / "post-cache-trace.svg")

    def test_post_cache_trace_summary_skips_exact_cache_boundary(self):
        processes = []
        for run in (1, 2):
            for batch in POST_CACHE.BATCHES:
                stages = []
                for index, name in enumerate(POST_CACHE.STAGES):
                    changed = batch > 1 and index == len(POST_CACHE.BASE.STAGES)
                    maximum = 0.01 if changed else 0.0
                    stages.append({
                        "name": name,
                        "b1_vs_batch_row0": {
                            "maximum": maximum, "rms": maximum / 2,
                            "relative_l2": maximum / 4,
                            "bitwise_equal": not changed},
                        "batch_row0_vs_row1": {
                            "maximum": 0.0, "rms": 0.0, "relative_l2": 0.0,
                            "bitwise_equal": True},
                    })
                processes.append({"batch": batch, "process_run": run,
                                  "stages": stages})
        summary = POST_CACHE.summarize(processes)
        self.assertTrue(summary["all_cache_cross_batch_bitwise_equal"])
        self.assertEqual(summary["first_nonzero_after_cache"],
                         POST_CACHE.PREFIX + ".attention.context")

    def test_current_fp32_qkv_model_gate_rejects_default(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-fp32-qkv-model-gate")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        policies = {row["policy"]: row for row in summary["policy_summaries"]}
        self.assertEqual(summary["precision_process_rows"], 16)
        self.assertEqual(summary["performance_process_rows"], 16)
        self.assertEqual(policies["invariant-qkv"]["cache_bitwise_case_count"], 4)
        self.assertEqual(
            policies["invariant-qkv"]["maximum_logit_cross_batch_error"],
            0.0012532472610473633)
        self.assertEqual(
            policies["invariant-qkv"]["maximum_logit_cross_batch_rms_error"],
            0.00029083847119404496)
        self.assertTrue(analysis["scoped_cache_fix_supported"])
        self.assertFalse(analysis["robust_complete_logit_improvement_supported"])
        self.assertEqual(verification["measurement_commit"],
                         "c34df680a3b25e4a806e44ad2b490afc461866fb")
        ET.parse(root / "model-gate.svg")

    def test_fp32_qkv_model_command_scopes_both_solutions(self):
        args = type("Args", (), {
            "binary": Path("micro"), "context": 8,
        })()
        model = {
            "config": "config.json", "weights": "model.bin",
            "inference": {"token_ids": [1, 2]},
        }
        default = FP32_QKV_MODEL.command(
            args, model, "default", 2, 0)
        candidate = FP32_QKV_MODEL.command(
            args, model, "invariant-qkv", 2, 0)
        self.assertNotIn("--fp32-prefill-q-solution-index", default)
        self.assertEqual(
            candidate[candidate.index("--fp32-prefill-q-solution-index") + 1],
            "296100")
        self.assertEqual(
            candidate[candidate.index("--fp32-prefill-kv-solution-index") + 1],
            "292135")

    def test_fp32_qkv_model_route_counts_include_warmup(self):
        base = {
            "status": "pass", "batch": 2, "token_count": 2048,
            "decode_tokens": 1, "kv_cache_dtype": "bf16",
            "cached_attention_materialized_policy": "auto-enabled",
            "fp32_prefill_q_solution_index": 296100,
            "fp32_prefill_kv_solution_index": 292135,
            "fp32_solution_registered_entries": 2,
            "fp32_solution_cached_algorithms": 2,
            "fp32_solution_cache_misses": 2,
            "fp32_solution_registry_hits": 168,
            "fp32_solution_cache_hits": 166,
            "fp32_solution_dispatches": 168,
        }
        FP32_QKV_MODEL.require_route(
            base, "invariant-qkv", 2, 2048, 1)
        base["fp32_solution_registry_hits"] = 84
        with self.assertRaises(ValueError):
            FP32_QKV_MODEL.require_route(
                base, "invariant-qkv", 2, 2048, 1)

    def test_fp32_qkv_model_summary_keeps_precision_and_performance_separate(self):
        metric = {"elements": 4, "maximum": 0.0, "rms": 0.0,
                  "bitwise_equal": True}
        precision = []
        performance = []
        for run in (1, 2):
            for policy in FP32_QKV_MODEL.POLICIES:
                for batch in FP32_QKV_MODEL.BATCHES:
                    precision.append({
                        "policy": policy, "batch": batch,
                        "key_cross_batch": metric, "value_cross_batch": metric,
                        "logits_cross_batch": metric, "key_vs_default": metric,
                        "value_vs_default": metric, "logits_vs_default": metric,
                        "key_within_batch_bitwise_equal": True,
                        "value_within_batch_bitwise_equal": True,
                        "logits_within_batch_bitwise_equal": True,
                        "host_device_argmax_equal": True,
                        "device_argmax_token": 3,
                    })
                    performance.append({
                        "policy": policy, "batch": batch,
                        "decode_prepare_ms": 10.0 if policy == "default" else 8.0,
                        "decode_tokens_per_second":
                            100.0 if policy == "default" else 110.0,
                        "engine_peak_bytes": 1000,
                        "generated_tokens": [3],
                    })
        summary = FP32_QKV_MODEL.summarize(precision, performance)
        self.assertEqual(summary["precision_process_rows"], 16)
        self.assertEqual(summary["performance_process_rows"], 16)
        self.assertEqual(summary["performance_cases"][0]["prefill_speedup"],
                         1.25)
        self.assertEqual(
            summary["performance_cases"][0]["decode_throughput_ratio"], 1.1)

    def test_current_fp32_qkv_row_invariance_selects_model_candidates(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-fp32-qkv-row-invariance")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        operations = {row["operation"]: row for row in summary["operations"]}
        self.assertEqual(operations["q"]["common_candidate_count"], 12)
        self.assertEqual(operations["q"]["block_invariant_count"], 1)
        self.assertEqual(operations["q"]["fastest_invariant_index"], 296100)
        self.assertEqual(operations["kv"]["common_candidate_count"], 22)
        self.assertEqual(operations["kv"]["block_invariant_count"], 5)
        self.assertEqual(operations["kv"]["fastest_invariant_index"], 292135)
        self.assertEqual(analysis["q_selected_workspace_bytes"], 0)
        self.assertEqual(analysis["kv_selected_workspace_bytes"], 0)
        self.assertEqual(verification["measurement_commit"],
                         "7a47f44c5f89266d38aca6feb2bce1b825e1d2b9")
        ET.parse(root / "qkv-row-invariance.svg")

    def test_fp32_qkv_summary_selects_fastest_invariant_candidate(self):
        def result(columns, rows):
            return {
                "common_candidate_count": len(rows),
                "supported_count": len(rows),
                "sentinel_pass_count": len(rows),
                "block_invariant_count": sum(
                    row["block_invariant"] for row in rows),
                "candidates": rows,
            }
        q = [
            {"index": 10, "block_invariant": False,
             "event_ms_p50": [1, 1, 1, 1], "maximum_workspace_bytes": 0,
             "block_maximum_error": 0.1, "sentinel_maximum_error": 0.0},
            {"index": 20, "block_invariant": True,
             "event_ms_p50": [2, 2, 2, 2], "maximum_workspace_bytes": 0,
             "block_maximum_error": 0.0, "sentinel_maximum_error": 0.0},
        ]
        kv = [
            {"index": 30, "block_invariant": True,
             "event_ms_p50": [1, 1, 1, 1], "maximum_workspace_bytes": 0,
             "block_maximum_error": 0.0, "sentinel_maximum_error": 0.0},
            {"index": 40, "block_invariant": True,
             "event_ms_p50": [0.5, 0.5, 0.5, 0.5],
             "maximum_workspace_bytes": 1024,
             "block_maximum_error": 0.0, "sentinel_maximum_error": 0.0},
        ]
        summary = FP32_QKV.summarize({
            "q": result(1536, q), "kv": result(256, kv),
        })
        operations = {row["operation"]: row for row in summary["operations"]}
        self.assertEqual(operations["q"]["fastest_invariant_index"], 20)
        self.assertEqual(operations["kv"]["fastest_invariant_index"], 40)
        self.assertEqual(operations["q"]["block_invariant_count"], 1)
        self.assertEqual(len(summary["candidates"]), 4)

    def test_current_prefill_block0_trace_locates_q_projection(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-deepseek-prefill-block0-trace")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        b8 = next(row for row in summary["cases"] if row["batch"] == 8)
        stages = {row["name"]: row for row in b8["stages"]}
        prefix = PREFILL_TRACE.PREFIX + ".attention."
        self.assertEqual(summary["process_rows"], 8)
        self.assertEqual(summary["stage_count"], 10)
        self.assertEqual(summary["first_nonzero_stage"],
                         prefix + "q_projection")
        self.assertEqual(stages[prefix + "q_projection"]
                         ["b1_vs_batch_row0"]["maximum"],
                         0.000091552734375)
        self.assertEqual(stages[prefix + "cache_key"]
                         ["b1_vs_batch_row0"]["maximum"], 0.03125)
        self.assertTrue(analysis["fp32_qkv_projection_is_first_source_supported"])
        self.assertEqual(verification["measurement_commit"],
                         "d861308481bf56e3590f66517a4ad68c1447e1ae")
        ET.parse(root / "prefill-trace.svg")

    def test_prefill_block0_trace_summary_finds_first_projection(self):
        processes = []
        for run in (1, 2):
            for batch in PREFILL_TRACE.BATCHES:
                stages = []
                for index, name in enumerate(PREFILL_TRACE.STAGES):
                    changed = batch > 1 and index >= 2
                    maximum = 0.01 if changed else 0.0
                    stages.append({
                        "name": name,
                        "b1_vs_batch_row0": {
                            "maximum": maximum, "rms": maximum / 2,
                            "relative_l2": maximum / 4,
                            "bitwise_equal": not changed},
                        "batch_row0_vs_row1": {
                            "maximum": 0.0, "rms": 0.0, "relative_l2": 0.0,
                            "bitwise_equal": True},
                    })
                processes.append({
                    "batch": batch, "process_run": run, "stages": stages,
                })
        summary = PREFILL_TRACE.summarize(processes)
        self.assertEqual(summary["process_rows"], 8)
        self.assertEqual(summary["stage_count"], 10)
        self.assertEqual(summary["first_nonzero_stage"],
                         PREFILL_TRACE.PREFIX + ".attention.q_projection")
        self.assertTrue(summary["all_repeat_metrics_equal"])

    def test_prefill_block0_trace_command_bounds_two_rows(self):
        args = type("Args", (), {
            "binary": Path("micro"), "context": 2048,
        })()
        model = {
            "config": "config.json", "weights": "model.bin",
            "inference": {"token_ids": [1, 2]},
        }
        command = PREFILL_TRACE.command(
            args, model, 8, Path("trace.jsonl"), Path("cache.bin"))
        self.assertEqual(command[command.index("--trace-max-elements") + 1],
                         str(2 * 2048 * 1536))
        self.assertEqual(command[command.index("--trace-value-filter") + 1],
                         ",".join(PREFILL_TRACE.STAGES))

    def test_current_prefill_cache_prefix_precedes_decode_drift(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-deepseek-prefill-cache-prefix")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        tensors = {row["tensor"]: row for row in summary["tensor_summaries"]}
        cases = {(row["tensor"], row["batch"]): row
                 for row in summary["cases"]}
        self.assertEqual(summary["process_rows"], 8)
        self.assertTrue(summary["all_repeat_bitwise_equal"])
        self.assertFalse(summary["all_within_batch_bitwise_equal"])
        self.assertEqual(tensors["key"]["maximum_cross_batch_error"], 0.03125)
        self.assertEqual(tensors["value"]["maximum_cross_batch_error"],
                         0.0009765625)
        self.assertTrue(cases[("key", 2)]["within_batch_bitwise_equal"])
        self.assertFalse(cases[("key", 8)]["within_batch_bitwise_equal"])
        self.assertTrue(analysis["prefill_cache_drift_present_before_decode"])
        self.assertEqual(verification["measurement_commit"],
                         "e0eda4591d0c58168d1ae32a819537d54128f6ea")
        ET.parse(root / "cache-prefix.svg")

    def test_prefill_cache_summary_separates_key_and_value(self):
        processes = []
        for run in (1, 2):
            for batch in PREFILL_CACHE.BATCHES:
                key = [[0.0, 1.0] for _ in range(batch)]
                value_row = [0.0, 2.0 + (0.25 if batch > 1 else 0.0)]
                value = [value_row for _ in range(batch)]
                processes.append({
                    "batch": batch, "process_run": run,
                    "values": {"key": key, "value": value},
                    "raw": {
                        "key": [bytes([0, 1])] * batch,
                        "value": [bytes([2, batch])] * batch,
                    },
                })
        summary = PREFILL_CACHE.summarize(processes)
        tensors = {row["tensor"]: row for row in summary["tensor_summaries"]}
        self.assertEqual(summary["process_rows"], 8)
        self.assertTrue(summary["all_repeat_bitwise_equal"])
        self.assertTrue(summary["all_within_batch_bitwise_equal"])
        self.assertEqual(tensors["key"]["maximum_cross_batch_error"], 0.0)
        self.assertAlmostEqual(tensors["value"]["maximum_cross_batch_error"],
                               0.25)

    def test_prefill_cache_bf16_decoder_preserves_raw_values(self):
        payload = struct.pack("<HHH", 0x3F80, 0xC000, 0x0000)
        self.assertEqual(PREFILL_CACHE.bf16_values(payload), [1.0, -2.0, 0.0])
        with self.assertRaises(ValueError):
            PREFILL_CACHE.bf16_values(b"x")

    def test_current_bf16_row_invariance_closes_gate_up_search(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-bf16-decode-row-invariance")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        self.assertEqual(summary["candidate_count"], 64)
        self.assertEqual(summary["supported_count"], 64)
        self.assertEqual(summary["reference_pass_count"], 64)
        self.assertEqual(summary["row_invariant_count"], 64)
        self.assertEqual(summary["maximum_reference_error"], 0.0)
        self.assertEqual(summary["maximum_row_error"], 0.0)
        self.assertIn(75892, summary["row_invariant_indices"])
        self.assertFalse(
            analysis["gate_up_intrinsic_identical_input_row_drift_supported"])
        self.assertEqual(verification["measurement_commit"],
                         "7fe8c75513d8fed4b670b697b89d59686376c363")
        ET.parse(root / "row-invariance.svg")

    def test_bf16_row_invariance_summary_joins_workspace_and_exactness(self):
        inventory = {
            "shapes": [{"candidates": [
                {"index": 10, "workspace_bytes": 0},
                {"index": 20, "workspace_bytes": 1024},
            ]}, {"candidates": [
                {"index": 10, "workspace_bytes": 2048},
                {"index": 20, "workspace_bytes": 1024},
            ]}],
        }
        matrix = {"candidates": [
            {"index": 10, "supported": True, "reference_passed": True,
             "row_invariant": True, "row_maximum_error": 0.0,
             "reference_maximum_error": 0.0},
            {"index": 20, "supported": True, "reference_passed": True,
             "row_invariant": False, "row_maximum_error": 0.25,
             "reference_maximum_error": 0.01},
        ]}
        summary = ROW_INVARIANCE.summarize(inventory, matrix)
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["row_invariant_count"], 1)
        self.assertEqual(summary["row_invariant_indices"], [10])
        self.assertEqual(summary["minimum_invariant_workspace_bytes"], 2048)
        self.assertEqual(summary["maximum_row_error"], 0.25)

    def test_current_bf16_decode_algorithm_rejects_common_solution(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-deepseek-bf16-decode-algorithm")
        inventory = json.loads((root / "inventory.json").read_text(
            encoding="utf-8"))
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        policies = {row["algorithm_policy"]: row
                    for row in summary["policy_summaries"]}
        cases = {(row["algorithm_policy"], row["batch"]): row
                 for row in summary["cases"]}
        self.assertEqual(inventory["common_candidate_count"], 64)
        self.assertTrue(all(row["candidate_count"] == 64
                            for row in inventory["shapes"]))
        self.assertIn(75892, inventory["common_indices"])
        self.assertEqual(summary["process_rows"], 16)
        self.assertTrue(summary["all_repeat_bitwise_equal"])
        self.assertEqual(policies["default"]["maximum_cross_batch_error"],
                         0.06298542022705078)
        self.assertEqual(
            policies["common-solution"]["maximum_cross_batch_error"],
            0.06993913650512695)
        self.assertGreater(
            cases[("common-solution", 8)]["cross_batch_maximum_rms_error"],
            cases[("default", 8)]["cross_batch_maximum_rms_error"])
        self.assertFalse(analysis["same_index_implies_same_reduction_tree_supported"])
        self.assertEqual(verification["measurement_commit"],
                         "863294404f533b3dec234866bbc647942f07027f")
        ET.parse(root / "algorithm.svg")

    def test_bf16_decode_algorithm_command_is_decode_only_and_explicit(self):
        args = type("Args", (), {
            "binary": Path("micro"), "context": 8, "warmup": 1,
            "algorithm_index": 75892,
        })()
        model = {
            "config": "config.json", "weights": "model.bin",
            "inference": {"token_ids": [1, 2]},
        }
        default = DECODE_ALGORITHM.command(
            args, model, "default", 2, Path("default.bin"))
        common = DECODE_ALGORITHM.command(
            args, model, "common-solution", 2, Path("common.bin"))
        self.assertNotIn("--bf16-decode-algorithm-index", default)
        self.assertEqual(
            common[common.index("--bf16-decode-algorithm-index") + 1],
            "75892")
        self.assertEqual(common[common.index("--workload") + 1], "decode")

    def test_bf16_decode_algorithm_summary_compares_complete_batches(self):
        measurements = []
        for run in (1, 2):
            for policy_index, policy in enumerate(DECODE_ALGORITHM.POLICIES):
                for batch in DECODE_ALGORITHM.BATCHES:
                    values = [0.0, 1.0, 2.0, 3.0]
                    if policy_index and batch > 1:
                        values = [0.0, 1.25, 2.0, 3.0]
                    measurements.append(({
                        "algorithm_policy": policy, "batch": batch,
                        "process_run": run, "within_batch_bitwise_equal": True,
                        "host_device_argmax_equal": True,
                        "device_argmax_token": 3,
                        "decode_tokens_per_second": 10.0 + batch,
                        "engine_peak_bytes": 1000 + policy_index,
                    }, values * batch))
        summary = DECODE_ALGORITHM.summarize(measurements, 4, 75892)
        policies = {row["algorithm_policy"]: row
                    for row in summary["policy_summaries"]}
        self.assertEqual(summary["process_rows"], 16)
        self.assertEqual(summary["case_rows"], 8)
        self.assertTrue(summary["all_repeat_bitwise_equal"])
        self.assertEqual(policies["common-solution"]["algorithm_index"], 75892)
        self.assertAlmostEqual(
            policies["common-solution"]["maximum_cross_batch_error"], 0.25)

    def test_current_bf16_ffn_layer_counterfactual_rejects_block_zero(self):
        root = (ROOT / "benchmarks/results" /
                "2026-08-26-deepseek-bf16-ffn-layer-counterfactual")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        policies = {row["precision_policy"]: row
                    for row in summary["policy_summaries"]}
        cases = {(row["precision_policy"], row["batch"]): row
                 for row in summary["cases"]}
        self.assertEqual(summary["process_rows"], 24)
        self.assertTrue(summary["all_repeat_bitwise_equal"])
        self.assertTrue(summary["all_host_device_argmax_equal"])
        self.assertFalse(summary["all_within_batch_bitwise_equal"])
        self.assertEqual(policies["bf16-all"]["converted_tensors"], 84)
        self.assertEqual(policies["bf16-except-block0"]["converted_tensors"], 81)
        self.assertEqual(policies["bf16-all"]["maximum_cross_batch_error"],
                         0.06298542022705078)
        self.assertEqual(
            policies["bf16-except-block0"]["maximum_cross_batch_error"],
            0.0569688081741333)
        self.assertGreater(
            cases[("bf16-except-block0", 8)]["cross_batch_maximum_error"],
            cases[("bf16-all", 8)]["cross_batch_maximum_error"])
        self.assertEqual(analysis["peak_bytes_added_by_block0_fp32"], 82575360)
        self.assertEqual(verification["measurement_commit"],
                         "985fe2a80c834379091db34655a8fcaae6b7f651")
        ET.parse(root / "counterfactual.svg")

    def test_bf16_ffn_layer_counterfactual_has_exact_conversion_contract(self):
        args = type("Args", (), {
            "binary": Path("micro"), "context": 8, "warmup": 1,
        })()
        model = {
            "config": "config.json", "weights": "model.bin",
            "inference": {"token_ids": [1, 2]},
        }
        command = LAYER_COUNTERFACTUAL.command(
            args, model, "bf16-except-block0", 2, Path("logits.bin"))
        self.assertEqual(command[command.index("--bf16-ffn") + 1], "true")
        self.assertEqual(command[command.index("--bf16-ffn-fp32-layers") + 1],
                         "0")
        self.assertEqual(LAYER_COUNTERFACTUAL.EXPECTED_CONVERTED,
                         {"fp32-linear": 0, "bf16-all": 84,
                          "bf16-except-block0": 81})

    def test_bf16_ffn_layer_counterfactual_summary_separates_policies(self):
        measurements = []
        for run in (1, 2):
            for policy_index, policy in enumerate(LAYER_COUNTERFACTUAL.POLICIES):
                for batch in LAYER_COUNTERFACTUAL.BATCHES:
                    values = [0.0, 1.0, 2.0, 3.0]
                    if policy_index and batch > 1:
                        values = [0.0, 1.0 + policy_index / 10.0, 2.0, 3.0]
                    measurements.append(({
                        "precision_policy": policy, "batch": batch,
                        "process_run": run, "within_batch_bitwise_equal": True,
                        "host_device_argmax_equal": True,
                        "device_argmax_token": 3,
                        "decode_tokens_per_second": 10.0 + batch,
                        "engine_peak_bytes": 1000 + policy_index,
                    }, values * batch))
        summary = LAYER_COUNTERFACTUAL.summarize(measurements, 4)
        policies = {row["precision_policy"]: row
                    for row in summary["policy_summaries"]}
        self.assertEqual(summary["process_rows"], 24)
        self.assertEqual(summary["case_rows"], 12)
        self.assertTrue(summary["all_repeat_bitwise_equal"])
        self.assertEqual(policies["bf16-except-block0"]["converted_tensors"], 81)
        self.assertAlmostEqual(
            policies["bf16-except-block0"]["maximum_cross_batch_error"], 0.2)

    def test_current_cached_block_detail_locates_bf16_input_cast(self):
        root = ROOT / "benchmarks/results/2026-08-26-deepseek-cached-block-detail"
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        policies = {row["precision_island"]: row
                    for row in summary["policy_summaries"]}
        ffn = {row["name"]: row for row in policies["bf16-ffn"]["stages"]}
        prefix = BLOCK_DETAIL.PREFIX + "."
        self.assertEqual(summary["process_rows"], 4)
        self.assertEqual(summary["first_hundredfold_bf16_ffn_stage"],
                         prefix + "ffn.gate")
        self.assertTrue(policies["bf16-ffn"]["all_b2_rows_bitwise_equal"])
        self.assertEqual(ffn[prefix + "attention.q_projection"]
                         ["b1_vs_b2_row0"]["maximum"], 0.0)
        self.assertEqual(ffn[prefix + "attention.context"]
                         ["b1_vs_b2_row0"]["maximum"],
                         0.000056160613894462585)
        self.assertEqual(ffn[prefix + "ffn.input_bf16"]
                         ["b1_vs_b2_row0"]["maximum"], 0.00048828125)
        self.assertEqual(ffn[prefix + "ffn.gate"]
                         ["b1_vs_b2_row0"]["maximum"], 0.0078125)
        self.assertEqual(analysis["first_low_precision_amplifier"],
                         prefix + "ffn.input_bf16")
        self.assertEqual(verification["measurement_commit"],
                         "cada6dfa8595a24c23575a0269d083ee0fd7744b")
        ET.parse(root / "block-detail.svg")

    def test_cached_block_detail_compares_rows_and_finds_material_stage(self):
        names = [
            BLOCK_DETAIL.PREFIX + ".attention_norm",
            BLOCK_DETAIL.PREFIX + ".ffn.input_bf16",
            BLOCK_DETAIL.PREFIX + ".ffn.gate",
        ]
        processes = []
        for policy in BLOCK_DETAIL.POLICIES:
            stages = []
            for index, name in enumerate(names):
                maximum = 1.0e-6
                relative = 1.0e-6
                if policy == "bf16-ffn" and index == 2:
                    maximum = 0.01
                    relative = 0.01
                stages.append({
                    "name": name,
                    "b1_vs_b2_row0": {
                        "maximum": maximum, "rms": maximum / 2,
                        "relative_l2": relative, "bitwise_equal": False},
                    "b2_row0_vs_row1": {
                        "maximum": 0.0, "rms": 0.0,
                        "relative_l2": 0.0, "bitwise_equal": True},
                })
            for run in (1, 2):
                processes.append({
                    "precision_island": policy,
                    "process_run": run,
                    "stages": stages,
                })
        summary = BLOCK_DETAIL.summarize(processes)
        policies = {row["precision_island"]: row
                    for row in summary["policy_summaries"]}
        self.assertEqual(summary["process_rows"], 4)
        self.assertEqual(summary["first_hundredfold_bf16_ffn_stage"], names[2])
        self.assertEqual(policies["bf16-ffn"]
                         ["first_stage_at_or_above_maximum_1e_3"], names[2])

    def test_cached_block_detail_command_is_scoped_to_block_zero(self):
        args = type("Args", (), {
            "binary": Path("micro"), "context": 8,
            "trace_max_elements": 200000,
        })()
        model = {
            "config": "config.json", "weights": "model.bin",
            "inference": {"token_ids": [1, 2]},
        }
        command = BLOCK_DETAIL.command(
            args, model, "fp32-linear", 1,
            Path("trace.jsonl"), Path("logits.bin"))
        self.assertEqual(command[command.index("--trace-all-layer-details") + 1],
                         "true")
        self.assertEqual(command[command.index("--trace-value-filter") + 1],
                         BLOCK_DETAIL.PREFIX)

    def test_current_cached_block_drift_locates_block_zero(self):
        root = ROOT / "benchmarks/results/2026-08-25-deepseek-cached-block-drift"
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        policies = {row["precision_island"]: row
                    for row in summary["policy_summaries"]}
        fp32 = {row["name"]: row for row in policies["fp32-linear"]["stages"]}
        ffn = {row["name"]: row for row in policies["bf16-ffn"]["stages"]}
        block0 = "inference.cached.blocks.0"
        self.assertEqual(summary["process_rows"], 4)
        self.assertEqual(summary["selected_stage_count"], 31)
        self.assertEqual(summary["first_tenfold_bf16_ffn_stage"], block0)
        self.assertEqual(fp32["inference.cached.embedding"]
                         ["b1_vs_b2_row0"]["maximum"], 0.0)
        self.assertEqual(fp32[block0]["b1_vs_b2_row0"]["maximum"],
                         0.000007621943950653076)
        self.assertEqual(ffn[block0]["b1_vs_b2_row0"]["maximum"],
                         0.003909111022949219)
        self.assertEqual(analysis["block27_bf16_ffn_maximum_error"],
                         0.5828399658203125)
        self.assertEqual(verification["measurement_commit"],
                         "1ba27b7d8a566d987be2f2dbf4a8d5dd2e67c64f")
        ET.parse(root / "block-drift.svg")

    def test_block_drift_summary_selects_first_tenfold_stage(self):
        processes = []
        for policy in BLOCK_DRIFT.POLICIES:
            stages = []
            for index, name in enumerate(BLOCK_DRIFT.ordered_names()):
                maximum = 0.001
                if policy == "bf16-ffn" and index >= 4:
                    maximum = 0.02
                stages.append({
                    "name": name,
                    "b1_vs_b2_row0": {
                        "maximum": maximum, "rms": maximum / 10,
                        "bitwise_equal": False},
                    "b2_row0_vs_row1": {
                        "maximum": 0.0, "rms": 0.0,
                        "bitwise_equal": True},
                })
            for run in (1, 2):
                processes.append({
                    "precision_island": policy, "process_run": run,
                    "stages": stages,
                })
        summary = BLOCK_DRIFT.summarize(processes)
        self.assertEqual(summary["process_rows"], 4)
        self.assertEqual(summary["selected_stage_count"], 31)
        self.assertEqual(summary["first_tenfold_bf16_ffn_stage"],
                         BLOCK_DRIFT.ordered_names()[4])

    def test_current_precision_isolation_selects_bf16_ffn(self):
        root = ROOT / "benchmarks/results/2026-08-25-deepseek-cross-batch-precision"
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        by_policy = {row["precision_island"]: row
                     for row in summary["policy_summaries"]}
        self.assertEqual(summary["process_rows"], 32)
        self.assertTrue(summary["all_repeat_bitwise_equal"])
        self.assertTrue(summary["all_host_device_argmax_equal"])
        self.assertEqual(by_policy["fp32-linear"]["maximum_cross_batch_error"],
                         0.0013535022735595703)
        self.assertEqual(by_policy["bf16-ffn"]["maximum_cross_batch_error"],
                         0.06298542022705078)
        self.assertEqual(
            by_policy["bf16-attention"]["maximum_cross_batch_error"],
            0.020970463752746582)
        self.assertEqual(verification["measurement_commit"],
                         "0a09653b84eb3655d16b9fd9b62b06202bfaac78")
        ET.parse(root / "precision-isolation.svg")

    def test_precision_isolation_summary_separates_weight_islands(self):
        measurements = []
        for run in (1, 2):
            for policy_index, policy in enumerate(PRECISION.POLICIES):
                for batch in (1, 2, 4, 8):
                    values = [0.0, 1.0, 2.0, 3.0]
                    if policy_index and batch > 1:
                        values = [0.0, 1.0 + policy_index / 10.0, 2.0, 3.0]
                    measurements.append(({
                        "precision_island": policy, "batch": batch,
                        "process_run": run,
                        "within_batch_bitwise_equal": True,
                        "host_device_argmax_equal": True,
                        "device_argmax_token": 3,
                        "engine_peak_bytes": 1000 + policy_index,
                    }, values * batch))
        summary = PRECISION.summarize(measurements, 4)
        self.assertEqual(summary["process_rows"], 32)
        self.assertEqual(summary["case_rows"], 16)
        self.assertTrue(summary["all_repeat_bitwise_equal"])
        self.assertTrue(summary["all_within_batch_bitwise_equal"])
        self.assertTrue(summary["all_host_device_argmax_equal"])
        by_policy = {row["precision_island"]: row
                     for row in summary["policy_summaries"]}
        self.assertEqual(by_policy["fp32-linear"]["maximum_cross_batch_error"], 0)
        self.assertAlmostEqual(
            by_policy["bf16-both"]["maximum_cross_batch_error"], 0.3)

    def test_current_cross_batch_audit_is_complete(self):
        root = ROOT / "benchmarks/results/2026-08-25-deepseek-cross-batch-logits"
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        raw = (root / "raw.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(raw), 24)
        self.assertEqual(summary["process_rows"], 24)
        self.assertTrue(summary["all_repeat_bitwise_equal"])
        self.assertTrue(summary["all_within_batch_bitwise_equal"])
        self.assertTrue(summary["all_host_device_argmax_equal"])
        self.assertFalse(summary["all_cross_batch_bitwise_equal"])
        self.assertEqual(summary["first_non_bitwise_step"], 0)
        self.assertEqual(summary["maximum_cross_batch_error"],
                         0.19780349731445312)
        self.assertEqual(verification["measurement_commit"],
                         "0203cd9cd0dbf68e0105bc6318c38f8aeb046d4e")
        ET.parse(root / "cross-batch.svg")

    def test_cross_batch_audit_finds_first_complete_logit_drift(self):
        measurements = []
        for run in (1, 2):
            for step in (0, 1, 2):
                for batch in (1, 2, 4, 8):
                    values = [0.0, 1.0, 2.0, 3.0]
                    if batch == 8 and step >= 1:
                        values = [0.0, 1.0, 2.25, 3.0]
                    logits = values * batch
                    measurements.append(({
                        "batch": batch, "decode_step": step,
                        "process_run": run,
                        "within_batch_bitwise_equal": True,
                        "host_device_argmax_equal": True,
                        "device_argmax_token": 3,
                    }, logits))
        summary = CROSS_BATCH.summarize(measurements, 4)
        self.assertEqual(summary["process_rows"], 24)
        self.assertEqual(summary["case_rows"], 12)
        self.assertTrue(summary["all_repeat_bitwise_equal"])
        self.assertTrue(summary["all_within_batch_bitwise_equal"])
        self.assertTrue(summary["all_host_device_argmax_equal"])
        self.assertFalse(summary["all_cross_batch_bitwise_equal"])
        self.assertEqual(summary["first_non_bitwise_step"], 1)
        self.assertAlmostEqual(summary["maximum_cross_batch_error"], 0.25)

    def test_current_serving_batch_scale_is_complete_and_rendered(self):
        root = ROOT / "benchmarks/results/2026-08-25-serving-batch-scale"
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        raw = [json.loads(line) for line in (root / "raw.jsonl").read_text(
            encoding="utf-8").splitlines() if line]
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(len(summary["rows"]), 8)
        self.assertEqual(len(raw), 48)
        self.assertEqual(sum(row["framework"] == "microllm" and
                             row["cached_attention_materialized_policy"] ==
                             "auto-enabled" for row in raw), 24)
        self.assertEqual(sum(row["framework"] == "pytorch" and
                             row["device_discovery_workaround"] ==
                             "amdsmi_zero_fallback_to_hip_runtime"
                             for row in raw), 24)
        self.assertEqual(analysis["qwen_b8_scaling"], 6.5851601491695755)
        self.assertEqual(analysis["deepseek_b8_scaling"], 6.282360624661898)
        self.assertEqual(analysis["deepseek_divergent_batches"], [1, 8])
        self.assertFalse(analysis["scheduler_default_admitted"])
        self.assertEqual(verification["measurement_commit"],
                         "b4138b9be073ae51407ab86263957c47d82a6dda")
        ET.parse(root / "batch-scale.svg")

    def test_current_deepseek_t2048_baseline_is_complete_and_exact(self):
        root = ROOT / "benchmarks/results/2026-08-25-current-deepseek-t2048"
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        attempts = json.loads((root / "attempts.json").read_text(encoding="utf-8"))
        raw = [json.loads(line) for line in (root / "raw.jsonl").read_text(
            encoding="utf-8").splitlines() if line]
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["runs_per_framework"], 3)
        self.assertEqual(len(summary["rows"]), 1)
        row = summary["rows"][0]
        self.assertEqual((row["context"], row["batch"], row["decode_tokens"]),
                         (2048, 2, 64))
        self.assertTrue(row["cross_framework_tokens_equal"])
        self.assertEqual(row["cross_framework_first_token_difference"], -1)
        self.assertEqual(row["cross_framework_matching_prefix_tokens"], 64)
        self.assertEqual(row["microllm_throughput_tokens_per_second"],
                         133.501430352)
        self.assertEqual(row["pytorch_throughput_tokens_per_second"],
                         163.64307901695054)
        self.assertEqual(row["throughput_ratio_microllm_over_pytorch"],
                         0.8158085948637742)
        self.assertEqual(row["microllm_kv_cache_actual_bytes"], 121110528.0)
        self.assertEqual(row["pytorch_kv_cache_actual_bytes"], 121110528.0)
        self.assertEqual(row["microllm_kv_cache_utilization"], 1.0)
        self.assertEqual(row["pytorch_kv_cache_utilization"], 1.0)
        self.assertEqual(len(raw), 6)
        self.assertTrue(all(record["status"] == "pass" for record in raw))
        self.assertEqual({record["framework"] for record in raw},
                         {"microllm", "pytorch"})
        self.assertEqual(verification["measurement_commit"],
                         "4ac239384a8c4f3f26b49e4884efeb32e0f02cfa")
        self.assertTrue(verification["performance_gap_reproduced"])
        self.assertEqual(attempts["attempts"][0]["status"],
                         "invalid_for_cross_framework_comparison")
        self.assertEqual(attempts["attempts"][1]["status"], "pass")

    def test_named_suites_cover_short_long_context_and_batch_scaling(self):
        self.assertEqual(MATRIX.MATRIX_SUITES["smoke"]["contexts"], [8, 128])
        standard = MATRIX.MATRIX_SUITES["standard"]
        self.assertIn(8, standard["contexts"])
        self.assertIn(2048, standard["contexts"])
        self.assertEqual(standard["batches"], [1, 2, 4, 8])
        serving = MATRIX.MATRIX_SUITES["serving"]
        self.assertEqual(serving["contexts"], [1, 8, 32, 128, 512, 2048])
        self.assertEqual(serving["batches"], [1, 2, 4, 8])
        self.assertEqual(serving["decode_lengths"], [1, 8, 32, 64])
        extended = MATRIX.MATRIX_SUITES["extended"]
        self.assertIn(1, extended["contexts"])
        self.assertIn(4096, extended["contexts"])
        self.assertIn(16, extended["batches"])
        self.assertEqual(extended["decode_lengths"], [1, 8, 32])
        boundary = MATRIX.MATRIX_SUITES["boundary"]
        self.assertTrue({31, 32, 33, 127, 128, 129, 511, 512, 513} <=
                        set(boundary["contexts"]))
        self.assertIn(3, boundary["batches"])

    def test_context_batch_and_kv_formulas_are_explicit(self):
        self.assertEqual(MATRIX.positive_int_list("8,128,512", "contexts"),
                         [8, 128, 512])
        self.assertEqual(MATRIX.expanded_tokens([10, 20, 30], 5),
                         [10, 20, 30, 10, 20])
        self.assertEqual(
            MATRIX.theoretical_kv_cache_bytes(
                layers=24, kv_heads=2, head_dimension=64,
                batch=4, tokens=512, element_bytes=2),
            2 * 24 * 2 * 64 * 4 * 512 * 2)
        with self.assertRaisesRegex(Exception, "unique positive"):
            MATRIX.positive_int_list("1,1", "batches")

    def test_validator_rejects_silent_batch_fallback(self):
        model = {"name": "qwen", "parameter_count": 10}
        record = {
            "status": "pass", "model": "qwen", "parameter_count": 10,
            "context": 128, "batch": 1, "workload": "decode",
            "cache_mode": "cached",
            "precision": "mixed_bf16_weights_fp32_activations", "warmup": 1,
            "steps": 2, "measured_tokens": 32, "peak_bytes": 100,
            "device_total_bytes": 1000, "resident_weight_bytes": 80,
            "kv_cache_active_bytes": 40, "kv_cache_utilization": 1.0,
            "throughput_tokens_per_second": 10.0,
            "kv_cache_theoretical_bytes": 40, "kv_cache_actual_bytes": 40,
        }
        with self.assertRaisesRegex(RuntimeError, "shape contract"):
            MATRIX.validate_measurement(
                record, model, "microllm", 128, 2, "decode", "cached", 1, 2, 8)

    def test_validator_rejects_prefill_semantic_drift(self):
        model = {"name": "qwen", "parameter_count": 10}
        record = {
            "status": "pass", "model": "qwen", "parameter_count": 10,
            "context": 128, "batch": 1, "workload": "prefill",
            "cache_mode": "uncached", "prefill_logits_mode": "full",
            "precision": "mixed_bf16_weights_fp32_activations",
            "warmup": 1, "steps": 2, "peak_bytes": 100,
            "device_total_bytes": 1000, "resident_weight_bytes": 80,
            "throughput_tokens_per_second": 10.0,
            "top_logits": [{"token": 1, "logit": 2.0}],
        }
        with self.assertRaisesRegex(RuntimeError, "prefill logits semantics"):
            MATRIX.validate_measurement(
                record, model, "microllm", 128, 1, "prefill", "uncached",
                1, 2, 8, "last")

    def test_summary_preserves_limits_and_computes_paired_medians(self):
        model = {"name": "qwen", "revision": "fixed"}
        records = []
        for framework, throughputs, peaks in (
            ("microllm", [100.0, 120.0, 110.0], [1000, 1200, 1100]),
            ("pytorch", [200.0, 220.0, 210.0], [800, 1000, 900]),
        ):
            for throughput, peak in zip(throughputs, peaks):
                records.append({
                    "model": "qwen", "context": 8, "batch": 1,
                    "workload": "prefill", "cache_mode": "uncached",
                    "framework": framework, "status": "pass",
                    "throughput_tokens_per_second": throughput,
                    "latency_ms": 1.0, "peak_bytes": peak,
                    "resident_weight_bytes": 700,
                    "kv_cache_actual_bytes": 0, "kv_cache_theoretical_bytes": 0,
                    "kv_cache_allocation_efficiency": 0.0,
                    "kv_cache_utilization": 0.0,
                    "kv_cache_share_of_peak": 0.0,
                })
        # The two decode cases remain explicitly limited because no fake samples exist.
        summary = MATRIX.summarize(records, [model], [8], [1], 3)
        prefill = summary["rows"][0]
        self.assertEqual(prefill["status"], "pass")
        self.assertEqual(prefill["microllm_throughput_tokens_per_second"], 110.0)
        self.assertEqual(prefill["pytorch_throughput_tokens_per_second"], 210.0)
        self.assertAlmostEqual(prefill["throughput_ratio_microllm_over_pytorch"],
                               110.0 / 210.0)
        self.assertAlmostEqual(prefill["peak_memory_ratio_microllm_over_pytorch"],
                               1100.0 / 900.0)
        self.assertEqual(summary["status"], "complete_with_recorded_limits")

    def test_failure_classification_keeps_unsupported_and_oom(self):
        self.assertEqual(MATRIX.classify_failure(
            "cached decode currently supports batch 1"), "unsupported")
        self.assertEqual(MATRIX.classify_failure("HIP out of memory"), "oom")
        self.assertEqual(MATRIX.classify_failure("wrong answer"), "failed")

    def test_first_sequence_difference_reports_exact_divergence(self):
        self.assertEqual(MATRIX.first_sequence_difference([1, 2], [1, 2]), -1)
        self.assertEqual(MATRIX.first_sequence_difference([1, 2], [1, 3]), 1)
        self.assertEqual(MATRIX.first_sequence_difference([1], [1, 2]), 1)

    def test_case_filter_rejects_unknown_and_omits_unrequested_rows(self):
        self.assertEqual(MATRIX.case_list("prefill,cached"), ["prefill", "cached"])
        with self.assertRaisesRegex(Exception, "cases must contain"):
            MATRIX.case_list("prefill,magic")
        model = {"name": "tiny", "revision": "fixed"}
        records = []
        for framework in ("microllm", "pytorch"):
            records.append({
                "model": "tiny", "context": 8, "batch": 1,
                "workload": "prefill", "cache_mode": "uncached",
                "framework": framework, "status": "pass",
                "throughput_tokens_per_second": 1.0, "latency_ms": 1.0,
                "peak_bytes": 10, "resident_weight_bytes": 8,
            })
        summary = MATRIX.summarize(
            records, [model], [8], [1], 1, cases=["prefill"])
        self.assertEqual(len(summary["rows"]), 1)
        self.assertEqual(summary["status"], "pass")

    def test_micro_normalization_separates_storage_and_active_cache(self):
        args = type("Args", (), {"warmup": 1, "steps": 2, "decode_tokens": 4,
                                  "micro_kv_cache_dtype": "bf16"})()
        raw = {
            "decode_tokens_per_second": 50.0, "mean_generation_ms": 8.0,
            "engine_peak_bytes": 2000, "measured_tokens": 8,
            "device_total_bytes": 10000, "engine_peak_share_of_device": 0.2,
            "resident_weight_bytes": 1000,
            "inference_weight_policy": "single_representation_bf16_ffn_attention",
            "kv_cache_actual_bytes": 320, "kv_cache_active_bytes": 224,
            "kv_cache_capacity_tokens": 10, "kv_cache_active_tokens": 7,
            "kv_cache_layers": 2, "kv_cache_heads": 1,
            "kv_cache_head_dimension": 4, "kv_cache_element_bytes": 2,
            "kv_cache_utilization": 0.7,
        }
        record = MATRIX.normalize_micro(
            raw, {"name": "tiny", "revision": "fixed"}, 6, 1,
            "decode", "cached", args)
        self.assertEqual(record["kv_cache_actual_bytes"], 320)
        self.assertEqual(record["kv_cache_theoretical_bytes"], 320)
        self.assertEqual(record["kv_cache_utilization"], 0.7)
        self.assertEqual(record["kv_cache_dtype"], "bf16")
        self.assertEqual(record["kv_cache_reservation_policy"],
                         "fixed_exact_capacity")
        self.assertEqual(record["peak_memory_share_of_device"], 0.2)
        self.assertEqual(record["precision"],
                         "mixed_bf16_weights_fp32_activations")

    def test_micro_command_propagates_explicit_bf16_cache_policy(self):
        args = type("Args", (), {
            "micro_binary": Path("micro"), "decode_tokens": 4,
            "warmup": 1, "steps": 2, "micro_batch_argmax_mode": "device",
            "micro_kv_cache_dtype": "bf16",
            "micro_kv_cache_fp32_layers": "0,2",
            "prefill_logits_mode": "last"})()
        model = {"config": "config.json", "weights": "weights.bin",
                 "inference": {"token_ids": [1, 2]}}
        command = MATRIX.micro_command(
            args, model, context=8, batch=4, workload="decode", cache="cached")
        policy = command.index("--kv-cache-dtype")
        self.assertEqual(command[policy + 1], "bf16")
        batch = command.index("--batch")
        self.assertEqual(command[batch + 1], "4")
        layers = command.index("--kv-cache-fp32-layers")
        self.assertEqual(command[layers + 1], "0,2")
        logits = command.index("--prefill-logits")
        self.assertEqual(command[logits + 1], "last")
        decode_mode = command.index("--decode-mode")
        self.assertEqual(command[decode_mode + 1], "steady")
        capacity = command.index("--cache-capacity")
        self.assertEqual(command[capacity + 1], "12")

    def test_mixed_layer_cache_theoretical_bytes_sum_each_dtype(self):
        args = type("Args", (), {"warmup": 1, "steps": 1,
                                  "decode_tokens": 4,
                                  "micro_kv_cache_dtype": "bf16"})()
        raw = {
            "decode_tokens_per_second": 10.0, "mean_generation_ms": 1.0,
            "engine_peak_bytes": 1000, "measured_tokens": 4,
            "device_total_bytes": 10000, "engine_peak_share_of_device": 0.1,
            "resident_weight_bytes": 800,
            "inference_weight_policy": "single_representation_bf16_ffn_attention",
            "kv_cache_actual_bytes": 192, "kv_cache_active_bytes": 192,
            "kv_cache_capacity_tokens": 4, "kv_cache_active_tokens": 4,
            "kv_cache_layers": 2, "kv_cache_heads": 1,
            "kv_cache_head_dimension": 4, "kv_cache_element_bytes": 0,
            "kv_cache_fp32_layers": 1, "kv_cache_bf16_layers": 1,
            "kv_cache_utilization": 1.0,
        }
        record = MATRIX.normalize_micro(
            raw, {"name": "tiny", "revision": "fixed"}, 4, 1,
            "decode", "cached", args)
        self.assertEqual(record["kv_cache_theoretical_bytes"], 192)
        self.assertEqual(record["kv_cache_fp32_layers"], 1)
        self.assertEqual(record["kv_cache_bf16_layers"], 1)

    def test_sweep_capacity_exposes_unused_short_decode_pages(self):
        args = type("Args", (), {
            "micro_binary": Path("micro"), "decode_tokens": 1,
            "decode_lengths": [1, 8], "micro_cache_capacity": "sweep-max",
            "warmup": 1, "steps": 2, "micro_batch_argmax_mode": "device",
            "micro_kv_cache_dtype": "bf16", "micro_kv_cache_fp32_layers": "",
            "prefill_logits_mode": "last"})()
        model = {"config": "config.json", "weights": "weights.bin",
                 "inference": {"token_ids": [1, 2]}}
        command = MATRIX.micro_command(
            args, model, context=32, batch=4, workload="decode", cache="cached",
            decode_tokens=1)
        capacity = command.index("--cache-capacity")
        self.assertEqual(command[capacity + 1], "40")

    def test_summary_computes_batch_and_memory_efficiency(self):
        model = {"name": "tiny", "revision": "fixed"}
        records = []
        for framework in ("microllm", "pytorch"):
            for batch, throughput, peak, cache in (
                    (1, 100.0, 1000.0, 100.0),
                    (4, 320.0, 2500.0, 400.0)):
                records.append({
                    "model": "tiny", "context": 128, "batch": batch,
                    "workload": "decode", "cache_mode": "cached",
                    "framework": framework, "status": "pass",
                    "throughput_tokens_per_second": throughput,
                    "latency_ms": 1.0, "peak_bytes": peak,
                    "device_total_bytes": 10000.0,
                    "peak_memory_share_of_device": peak / 10000.0,
                    "resident_weight_bytes": 800.0,
                    "kv_cache_actual_bytes": cache,
                    "kv_cache_theoretical_bytes": cache,
                    "kv_cache_allocation_efficiency": 1.0,
                    "kv_cache_utilization": 1.0,
                    "kv_cache_share_of_peak": cache / peak,
                    "generated_tokens": [1, 2],
                })
        summary = MATRIX.summarize(
            records, [model], [128], [1, 4], 1, cases=["cached"])
        row = summary["rows"][1]
        self.assertAlmostEqual(row["microllm_batch_throughput_scaling"], 3.2)
        self.assertAlmostEqual(row["microllm_batch_efficiency"], 0.8)
        self.assertAlmostEqual(row["microllm_peak_memory_scaling"], 2.5)
        self.assertEqual(row["microllm_kv_cache_bytes_per_request"], 100.0)
        self.assertEqual(row["microllm_peak_bytes_per_request"], 625.0)
        self.assertEqual(summary["axes"]["batches"], [1, 4])

    def test_validator_checks_active_capacity_and_utilization(self):
        model = {"name": "tiny", "parameter_count": 10}
        record = {
            "status": "pass", "model": "tiny", "parameter_count": 10,
            "context": 8, "batch": 2, "workload": "decode",
            "cache_mode": "cached",
            "precision": "mixed_bf16_weights_fp32_activations",
            "warmup": 1, "steps": 2, "measured_tokens": 16,
            "measured_forward_steps": 16,
            "peak_bytes": 1000, "device_total_bytes": 10000,
            "resident_weight_bytes": 800,
            "throughput_tokens_per_second": 20.0,
            "decode_step_semantics": "one_model_forward_per_measured_token",
            "kv_cache_theoretical_bytes": 120,
            "kv_cache_actual_bytes": 120, "kv_cache_active_bytes": 120,
            "kv_cache_capacity_tokens": 12, "kv_cache_active_tokens": 12,
            "kv_cache_utilization": 1.0,
        }
        MATRIX.validate_measurement(
            record, model, "microllm", 8, 2, "decode", "cached", 1, 2, 4)
        record["kv_cache_active_tokens"] = 11
        with self.assertRaisesRegex(RuntimeError, "KV token accounting"):
            MATRIX.validate_measurement(
                record, model, "microllm", 8, 2, "decode", "cached", 1, 2, 4)

    def test_decode_length_axis_and_process_tail_metrics_are_preserved(self):
        model = {"name": "tiny", "revision": "fixed"}
        records = []
        for decode_tokens in (1, 8):
            for framework in ("microllm", "pytorch"):
                for run, latency in enumerate((10.0, 12.0, 20.0), start=1):
                    records.append({
                        "model": "tiny", "context": 128, "batch": 2,
                        "decode_tokens": decode_tokens,
                        "workload": "decode", "cache_mode": "cached",
                        "framework": framework, "status": "pass",
                        "process_run": run,
                        "throughput_tokens_per_second":
                            decode_tokens * 2 * 1000.0 / latency,
                        "latency_ms": latency, "peak_bytes": 1200,
                        "device_total_bytes": 10000,
                        "peak_memory_share_of_device": 0.12,
                        "resident_weight_bytes": 800,
                        "kv_cache_actual_bytes": 240,
                        "kv_cache_active_bytes": 220,
                        "kv_cache_theoretical_bytes": 240,
                        "kv_cache_capacity_tokens": 136,
                        "kv_cache_active_tokens": 135,
                        "kv_cache_element_bytes": 2,
                        "kv_cache_allocation_efficiency": 1.0,
                        "kv_cache_utilization": 220 / 240,
                        "kv_cache_share_of_peak": 0.2,
                        "generated_tokens": list(range(decode_tokens)),
                    })
        summary = MATRIX.summarize(
            records, [model], [128], [2], 3, cases=["cached"],
            decode_lengths=[1, 8])
        self.assertEqual(len(summary["rows"]), 2)
        self.assertEqual(summary["axes"]["decode_lengths"], [1, 8])
        long_row = summary["rows"][1]
        self.assertEqual(long_row["decode_tokens"], 8)
        self.assertEqual(long_row["microllm_process_latency_ms_p50"], 12.0)
        self.assertAlmostEqual(long_row["microllm_process_latency_ms_p95"], 19.2)
        self.assertEqual(long_row["microllm_peak_incremental_bytes"], 400.0)
        self.assertEqual(long_row[
            "microllm_peak_incremental_bytes_per_request"], 200.0)
        self.assertAlmostEqual(long_row[
            "microllm_kv_cache_share_of_incremental_peak"], 0.6)
        self.assertEqual(long_row["microllm_kv_cache_waste_bytes"], 20.0)
        self.assertEqual(long_row[
            "microllm_kv_cache_waste_bytes_per_request"], 10.0)
        self.assertAlmostEqual(long_row["microllm_kv_cache_waste_ratio"], 1.0 / 12.0)
        self.assertAlmostEqual(long_row[
            "microllm_kv_cache_active_share_of_incremental_peak"], 0.55)
        self.assertEqual(long_row["microllm_non_kv_incremental_bytes"], 160.0)
        self.assertGreater(long_row[
            "microllm_decode_length_throughput_vs_shortest"], 1.0)


if __name__ == "__main__":
    unittest.main()
