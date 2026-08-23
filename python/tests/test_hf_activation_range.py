import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks/single_gpu"
sys.path.insert(0, str(BENCHMARKS))
SPEC = importlib.util.spec_from_file_location(
    "hf_activation_range", BENCHMARKS / "hf_activation_range.py")
RANGE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RANGE)


class HfActivationRangeTest(unittest.TestCase):
    def test_selects_every_linear_input_and_marks_range_overflow(self):
        records = []
        for layer in range(2):
            for boundary in RANGE.BOUNDARIES:
                records.append({
                    "name": f"inference.blocks.{layer}.{boundary}",
                    "shape": [1, 2], "dtype": "float32",
                    "statistics": {"numel": 2, "finite_count": 2,
                                   "minimum": -10.0, "maximum": 60.0},
                })
        model = {"name": "tiny", "revision": "r", "layers": 2}
        selected = RANGE.select_ranges(records, model, 0.2)
        self.assertEqual(len(selected), 8)
        self.assertTrue(all(row["potential_saturation"] for row in selected))
        self.assertEqual(selected[0]["representable_magnitude"], 48.0)

    def test_rejects_missing_or_nonfinite_boundary(self):
        model = {"name": "tiny", "revision": "r", "layers": 1}
        with self.assertRaises(RuntimeError):
            RANGE.select_ranges([], model, 0.2)
        records = [{
            "name": f"inference.blocks.0.{boundary}",
            "shape": [1], "dtype": "float32",
            "statistics": {"numel": 1, "finite_count": 0,
                           "minimum": 0.0, "maximum": 0.0},
        } for boundary in RANGE.BOUNDARIES]
        with self.assertRaises(RuntimeError):
            RANGE.select_ranges(records, model, 0.2)

    def test_command_requests_all_layers_but_serializes_one_value(self):
        args = type("Args", (), {"binary": Path("micro"), "context": 8})()
        model = {"config": "config.json", "weights": "model.bin",
                 "inference": {"token_ids": [1, 2]}}
        command = RANGE.command(args, model, Path("trace.jsonl"))
        self.assertEqual(command[command.index("--trace-all-layer-details") + 1],
                         "true")
        self.assertEqual(command[command.index("--trace-max-elements") + 1], "1")


if __name__ == "__main__":
    unittest.main()
