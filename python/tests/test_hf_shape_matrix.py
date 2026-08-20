import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "hf_training_shape_matrix",
    ROOT / "benchmarks/single_gpu/hf_training_shape_matrix.py")
MATRIX = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MATRIX)


class HfTrainingShapeMatrixTest(unittest.TestCase):
    def test_shapes_and_token_expansion_are_explicit(self):
        self.assertEqual(MATRIX.parse_shapes("1x3,2x8"), [(1, 3), (2, 8)])
        self.assertEqual(MATRIX.expanded_tokens("10,20,30", 5), "10,20,30,10,20,30")
        with self.assertRaisesRegex(Exception, "duplicates"):
            MATRIX.parse_shapes("1x3,1x3")

    def test_summary_uses_per_framework_medians(self):
        model = {"name": "qwen", "revision": "fixed"}
        records = []
        for framework, throughputs, peaks in (
            ("microllm", [10.0, 14.0, 12.0], [100, 120, 110]),
            ("pytorch", [8.0, 10.0, 9.0], [80, 100, 90]),
        ):
            for run, (throughput, peak) in enumerate(zip(throughputs, peaks), 1):
                records.append({"model": "qwen", "batch": 2, "context": 8,
                                "status": "pass", "framework": framework,
                                "process_run": run, "tokens_per_second": throughput,
                                "peak_bytes": peak, "final_loss": 1.0})
        summary = MATRIX.summarize(records, [model], [(2, 8)], "bf16", 3)
        row = summary["rows"][0]
        self.assertEqual(row["microllm_tokens_per_second"], 12.0)
        self.assertEqual(row["pytorch_tokens_per_second"], 9.0)
        self.assertAlmostEqual(row["throughput_ratio_microllm_over_pytorch"], 4.0 / 3.0)
        self.assertAlmostEqual(row["peak_memory_ratio"], 110.0 / 90.0)

    def test_validation_rejects_silent_batch_fallback(self):
        model = {"name": "qwen", "parameter_count": 10}
        record = {"status": "pass", "parameter_count": 10, "batch": 1,
                  "context": 8, "warmup": 1, "steps": 2, "trained_tokens": 16,
                  "tokens_per_second": 1.0, "mean_step_ms": 1.0, "loss": 1.0,
                  "parameter_changed": True, "engine_peak_bytes": 100}
        with self.assertRaisesRegex(RuntimeError, "shape contract"):
            MATRIX.validate_record(record, model, "microllm", 2, 8, 1, 2)


if __name__ == "__main__":
    unittest.main()
