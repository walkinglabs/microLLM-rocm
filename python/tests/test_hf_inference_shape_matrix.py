import importlib.util
import json
import struct
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


class HfInferenceShapeMatrixTest(unittest.TestCase):
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
