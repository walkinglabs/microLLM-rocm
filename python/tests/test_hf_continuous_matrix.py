import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "hf_continuous_matrix",
    ROOT / "benchmarks/single_gpu/hf_continuous_matrix.py")
MATRIX = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MATRIX)
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

    def test_cache_formula_uses_request_bound_not_model_maximum(self):
        case = {"slots": 2, "prompts": [8, 32], "outputs": [4, 6]}
        layers, kv_heads, head_dimension = MATRIX.model_cache_shape(
            self.model()["config"])
        self.assertEqual(
            MATRIX.theoretical_cache_bytes(self.model(), case),
            2 * layers * kv_heads * head_dimension * 2 * 38 * 2)

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


if __name__ == "__main__":
    unittest.main()
