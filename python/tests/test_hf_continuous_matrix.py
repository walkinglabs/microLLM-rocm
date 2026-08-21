import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "hf_continuous_matrix",
    ROOT / "benchmarks/single_gpu/hf_continuous_matrix.py")
MATRIX = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MATRIX)
sys.modules["hf_continuous_matrix"] = MATRIX
TORCH_SPEC = importlib.util.spec_from_file_location(
    "pytorch_continuous_reference",
    ROOT / "benchmarks/single_gpu/pytorch_continuous_reference.py")
TORCH_REFERENCE = importlib.util.module_from_spec(TORCH_SPEC)
assert TORCH_SPEC.loader is not None
TORCH_SPEC.loader.exec_module(TORCH_REFERENCE)
COMPARE_SPEC = importlib.util.spec_from_file_location(
    "compare_hf_continuous",
    ROOT / "benchmarks/single_gpu/compare_hf_continuous.py")
COMPARE = importlib.util.module_from_spec(COMPARE_SPEC)
assert COMPARE_SPEC.loader is not None
COMPARE_SPEC.loader.exec_module(COMPARE)
DIVERGENCE_SPEC = importlib.util.spec_from_file_location(
    "hf_continuous_divergence",
    ROOT / "benchmarks/single_gpu/hf_continuous_divergence.py")
DIVERGENCE = importlib.util.module_from_spec(DIVERGENCE_SPEC)
assert DIVERGENCE_SPEC.loader is not None
DIVERGENCE_SPEC.loader.exec_module(DIVERGENCE)
ROW_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "hf_prefill_row_audit",
    ROOT / "benchmarks/single_gpu/hf_prefill_row_audit.py")
ROW_AUDIT = importlib.util.module_from_spec(ROW_AUDIT_SPEC)
assert ROW_AUDIT_SPEC.loader is not None
ROW_AUDIT_SPEC.loader.exec_module(ROW_AUDIT)


class HfContinuousMatrixTest(unittest.TestCase):
    def model(self):
        return {
            "name": "qwen",
            "revision": "fixed",
            "config": str(ROOT / "tests/fixtures/qwen25-0.5b-config.json"),
            "weights": "/models/qwen.safetensors",
            "inference": {"token_ids": [1, 2]},
        }

    def test_standard_suite_covers_short_long_slots_and_refill(self):
        standard = MATRIX.SUITES["standard"]
        self.assertEqual(set(standard),
                         {"short_s2", "short_s4", "long_s2", "long_s4"})
        self.assertEqual(standard["short_s2"]["slots"], 2)
        self.assertEqual(standard["long_s4"]["slots"], 4)
        self.assertIn(2048, standard["long_s2"]["prompts"])
        self.assertGreater(len(standard["long_s4"]["prompts"]),
                           standard["long_s4"]["slots"])

    def test_slot_sweep_holds_requests_fixed_across_one_two_four_eight_slots(self):
        sweep = MATRIX.SUITES["slot-sweep"]
        self.assertEqual({case["slots"] for case in sweep.values()}, {1, 2, 4, 8})
        for group in ("short", "long"):
            cases = [case for case in sweep.values() if case["group"] == group]
            self.assertEqual(len(cases), 4)
            self.assertEqual(len({tuple(case["prompts"]) for case in cases}), 1)
            self.assertEqual(len({tuple(case["outputs"]) for case in cases}), 1)
            self.assertTrue(all(len(case["prompts"]) == 8 for case in cases))

    def test_cache_formula_uses_request_bound_not_model_maximum(self):
        case = {"slots": 2, "prompts": [8, 32], "outputs": [4, 6]}
        layers, kv_heads, head_dimension = MATRIX.model_cache_shape(
            self.model()["config"])
        self.assertEqual(
            MATRIX.theoretical_cache_bytes(self.model(), case),
            2 * layers * kv_heads * head_dimension * 2 * 38 * 2)

    def test_token_difference_keeps_accuracy_failure_as_data(self):
        exact = MATRIX.token_difference([[1, 2], [3]], [[1, 2], [3]])
        self.assertTrue(exact["exact"])
        changed = MATRIX.token_difference([[1, 2], [3, 4]],
                                          [[1, 2], [3, 5]])
        self.assertFalse(changed["exact"])
        self.assertEqual(changed["differing_requests"], [1])
        self.assertEqual(changed["first_difference"], {"request": 1, "token": 1})

    def test_command_and_validator_preserve_exact_axes(self):
        case = {"slots": 2, "prompts": [8, 32], "outputs": [4, 6]}
        command = MATRIX.command(Path("micro"), self.model(), case, 1, 3)
        self.assertEqual(command[command.index("--workload") + 1], "continuous")
        self.assertEqual(command[command.index("--continuous-slots") + 1], "2")
        expected_cache = MATRIX.theoretical_cache_bytes(self.model(), case)
        record = {
            "status": "pass",
            "record_type": "official_continuous_serving_measurement",
            "request_count": 2,
            "continuous_slots": 2,
            "warmup": 1,
            "steps": 3,
            "measured_tokens": 30,
            "prompt_lengths": [8, 32],
            "new_token_lengths": [4, 6],
            "deterministic_across_steps": True,
            "allocated_cache_bytes": expected_cache,
            "peak_active_cache_bytes": expected_cache // 2,
            "kv_cache_byte_utilization": 0.5,
            "tokens_per_second": 10.0,
            "engine_peak_bytes": 1000,
            "resident_weight_bytes": 800,
        }
        normalized = MATRIX.validate(
            record, self.model(), "case", case, 1, 3)
        self.assertEqual(normalized["model"], "qwen")
        record["allocated_cache_bytes"] += 1
        with self.assertRaisesRegex(RuntimeError, "memory/timing"):
            MATRIX.validate(record, self.model(), "case", case, 1, 3)

    def test_pytorch_reference_is_explicitly_sequential_and_checksum_matches(self):
        self.assertEqual(TORCH_REFERENCE.positive_list("1,2,3", "values"),
                         [1, 2, 3])
        self.assertEqual(TORCH_REFERENCE.checksum([[1, 2], [3]]),
                         ((1 * 131 + 2) * 131 + 3))
        self.assertIn("sequential-request", TORCH_REFERENCE.__doc__.lower())
        self.assertIn("without pretending", TORCH_REFERENCE.__doc__.lower())

    def test_comparison_marks_token_mismatch_and_names_boundary(self):
        base = {
            "model": "qwen", "case": "short_s2", "status": "pass",
            "request_count": 1, "continuous_slots": 1,
            "prompt_lengths": [8], "new_token_lengths": [2],
            "generated_tokens": [[3, 4]], "token_checksum": 397,
            "allocated_cache_bytes": 100, "peak_active_cache_bytes": 80,
            "kv_cache_byte_utilization": 0.8, "slot_utilization": 1.0,
            "resident_weight_bytes": 500, "engine_peak_bytes": 700,
        }
        micro = {
            "track": "official_continuous_serving_matrix", "status": "pass",
            "runs": 2, "rows": [{**base, "process_run": 1},
                                  {**base, "process_run": 2}],
            "aggregates": [{
                "model": "qwen", "case": "short_s2", "status": "pass",
                "successful_runs": 2, "tokens_per_second_p50": 20.0,
                "tokens_per_second_min": 19.0, "tokens_per_second_max": 21.0,
            }],
        }
        pytorch = {("qwen", "short_s2"): {
            "prompt_lengths": [8], "new_token_lengths": [2],
            "generated_tokens": [[3, 5]], "tokens_per_second": 10.0,
            "peak_bytes": 600, "resident_weight_bytes": 400,
        }}
        result = COMPARE.compare(micro, pytorch)
        self.assertEqual(result["status"],
                         "complete_with_recorded_accuracy_failures")
        self.assertEqual(result["rows"][0]["accuracy_status"], "fail")
        self.assertEqual(result["rows"][0]["observed_service_throughput_ratio"], 2.0)
        self.assertIn("sequential requests", result["comparison_boundary"])

    def test_divergence_summary_keeps_top2_source_and_margin(self):
        model = self.model()
        default_command = DIVERGENCE.command(
            Path("micro"), model, DIVERGENCE.CASES["short_s4"])
        serial_command = DIVERGENCE.command(
            Path("micro"), model,
            DIVERGENCE.CASES["short_s4_serial_prefill"])
        self.assertEqual(
            default_command[default_command.index("--continuous-diagnostics") + 1],
            "true")
        self.assertNotIn("--continuous-prefill-batch", default_command)
        self.assertEqual(
            serial_command[serial_command.index("--continuous-prefill-batch") + 1],
            "false")
        records = []
        for case_name, case in DIVERGENCE.CASES.items():
            slots = case["slots"]
            changed = slots >= 4 and case["batch_equal_length_prefill"]
            generated = [[1, 2] for _ in range(5)] + [
                [1, 3 if changed else 2]]
            diagnostics = [
                {"request_id": 6, "generated_index": 0,
                 "device_argmax_matches_top1": True},
                {"request_id": 6, "generated_index": 1,
                 "device_selected_token": generated[5][1],
                 "top1_token": generated[5][1], "top1_logit": 5.0,
                 "top2_token": 2 if changed else 3, "top2_logit": 4.99,
                 "top1_top2_margin": 0.01, "logit_source": "uniform_decode",
                 "logit_batch_size": slots, "cache_position": 9,
                 "scheduler_step": 2, "device_argmax_matches_top1": True},
            ]
            records.append({
                "case": case_name, "status": "pass",
                "generated_tokens": generated,
                "selection_diagnostic_count": sum(
                    case["outputs"]),
                "selection_diagnostics": diagnostics,
            })
            missing = sum(case["outputs"]) - 2
            records[-1]["selection_diagnostics"].extend(
                {"request_id": 99, "generated_index": index,
                 "device_argmax_matches_top1": True}
                for index in range(missing))
        summary = DIVERGENCE.summarize(records, "deepseek", 1)
        self.assertEqual(summary["first_difference"], {"request": 5, "token": 1})
        self.assertEqual(summary["diagnostic_evidence"][2]["top2_tokens"], [2])
        self.assertEqual(summary["diagnostic_evidence"][2]["logit_batch_sizes"], [4])
        serial_s4 = next(row for row in summary["comparisons"]
                         if row["case"] == "short_s4_serial_prefill")
        self.assertTrue(serial_s4["difference_vs_s1"]["exact"])
        self.assertIn("excluded", summary["measurement_boundary"])
        reference = {"serving_mode": "sequential_requests",
                     "precision": "full_bf16_model",
                     "generated_tokens": [[1, 2] for _ in range(5)] + [[1, 3]]}
        comparison = DIVERGENCE.compare_to_pytorch(records, reference)
        self.assertTrue(
            comparison["default_s4_matches_reference_at_original_divergence"])
        self.assertFalse(
            comparison["serial_s4_matches_reference_at_original_divergence"])
        self.assertIn("not a matched scheduler", comparison["boundary"])

    def test_prefill_row_audit_command_preserves_explicit_offsets(self):
        model = self.model()
        case = ROW_AUDIT.CASES["pair_5_4"]
        command = ROW_AUDIT.command(Path("micro"), model, case)
        self.assertEqual(
            command[command.index("--continuous-prompt-offsets") + 1], "5,4")
        self.assertEqual(case["targets"], [0])
        self.assertEqual(ROW_AUDIT.CASES["duplicate_5"]["offsets"], [5, 5])
        first = {"device_selected_token": 3, "top1_token": 3,
                 "top1_logit": 2.0, "top2_token": 4, "top2_logit": 1.0,
                 "top1_top2_margin": 1.0, "cache_position": 32,
                 "logit_batch_size": 2, "logit_source": "prefill"}
        self.assertEqual(ROW_AUDIT.numeric_signature(first),
                         ROW_AUDIT.numeric_signature(dict(first)))


if __name__ == "__main__":
    unittest.main()
