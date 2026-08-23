import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks/single_gpu"
sys.path.insert(0, str(BENCHMARKS))
SPEC = importlib.util.spec_from_file_location(
    "hf_activation_row_range", BENCHMARKS / "hf_activation_row_range.py")
ROWS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ROWS)


class HfActivationRowRangeTest(unittest.TestCase):
    def test_row_range_exposes_within_tensor_skew(self):
        record = {
            "name": "inference.blocks.3.ffn.activated", "shape": [2, 2],
            "values": [1.0, -2.0, 8.0, -4.0], "values_truncated": False,
        }
        row = ROWS.row_range(record, {"name": "m", "revision": "r"})
        self.assertEqual(row["row_amax"], [2.0, 8.0])
        self.assertEqual(row["row_spread"], 4.0)
        self.assertEqual(row["rows_at_or_below_quarter_tensor_amax"], 1)

    def test_rejects_truncated_selected_values(self):
        record = {
            "name": "inference.blocks.0.attention_norm", "shape": [2, 2],
            "values": [1.0], "values_truncated": True,
        }
        with self.assertRaises(RuntimeError):
            ROWS.row_range(record, {"name": "m", "revision": "r"})

    def test_command_filters_four_boundaries_and_keeps_complete_values(self):
        args = type("Args", (), {"binary": Path("micro"), "context": 8})()
        model = {"config": "config.json", "weights": "model.bin",
                 "inference": {"token_ids": [1, 2]}}
        command = ROWS.command(args, model, Path("trace.jsonl"))
        filters = command[command.index("--trace-value-filter") + 1]
        self.assertEqual(filters.split(","), list(ROWS.BOUNDARIES))
        self.assertEqual(command[command.index("--trace-max-elements") + 1],
                         "1000000")


if __name__ == "__main__":
    unittest.main()
